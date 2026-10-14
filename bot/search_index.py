"""チャンネル全体の検索インデックス (SQLite + 埋め込みベクトル) とバックグラウンド索引付け.

- 発言は「会話のまとまり」(チャンク: 時間的に連続した数件〜十数件の発言) ごとに
  sarashina3-embedding でベクトル化する。1発言ずつだと短すぎて意味が取りにくく、
  埋め込み API の 60 RPM 制限にもすぐ当たるため。
- ベクトルは先頭256次元に切り詰めて正規化して保存する (Matryoshka 表現。精度はほぼ同等で軽い)。
- 発言本文は SQLite に保存し、キーワード検索 (LIKE) と結果表示・リンク生成に使う。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import discord
import numpy as np

from sarashina import EMBED_MODEL, SarashinaClient

log = logging.getLogger("sarashina-bot.index")

EMBED_DIM = 256
CHUNK_MAX_MESSAGES = 12
CHUNK_MAX_CHARS = 1200
CHUNK_GAP = timedelta(minutes=15)  # これ以上間が空いたら別の会話とみなす

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    author_name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    chunk_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_chunk ON messages(chunk_id);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id, created_at);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    start_at REAL NOT NULL,
    end_at REAL NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS channel_state (
    channel_id INTEGER PRIMARY KEY,
    newest_id INTEGER,
    oldest_id INTEGER,
    backfilled INTEGER NOT NULL DEFAULT 0,
    indexed_messages INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class StoredMessage:
    id: int
    guild_id: int
    channel_id: int
    channel_name: str
    author_id: int
    author_name: str
    content: str
    created_at: float
    chunk_id: int | None

    @property
    def jump_url(self) -> str:
        return f"https://discord.com/channels/{self.guild_id}/{self.channel_id}/{self.id}"

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.created_at).strftime("%Y/%m/%d %H:%M")


def keyword_pattern(keyword: str) -> re.Pattern:
    """英数字だけのキーワードは単語単位、それ以外は部分一致の正規表現."""
    escaped = re.escape(keyword)
    if re.fullmatch(r"[A-Za-z0-9_.+-]+", keyword):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def normalize(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec[:EMBED_DIM], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def message_text(m: discord.Message) -> str:
    text = m.clean_content.strip()
    if m.attachments:
        text += " " + " ".join(f"[添付: {a.filename}]" for a in m.attachments)
    return text.strip()


def split_chunks(messages: list[discord.Message]) -> list[list[discord.Message]]:
    """時系列順の発言を、会話のまとまりごとに分ける."""
    chunks: list[list[discord.Message]] = []
    current: list[discord.Message] = []
    chars = 0
    for m in messages:
        text_len = len(message_text(m))
        if current and (
            m.created_at - current[-1].created_at > CHUNK_GAP
            or len(current) >= CHUNK_MAX_MESSAGES
            or chars + text_len > CHUNK_MAX_CHARS
        ):
            chunks.append(current)
            current, chars = [], 0
        current.append(m)
        chars += text_len
    if current:
        chunks.append(current)
    return chunks


class SearchIndex:
    """SQLite に発言とチャンクを保存し、ベクトルはメモリ上の行列で検索する."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self._load_vectors()

    def _load_vectors(self) -> None:
        rows = self.db.execute(
            "SELECT id, channel_id, start_at, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        self.chunk_ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.chunk_channels = np.array([r[1] for r in rows], dtype=np.int64)
        self.chunk_times = np.array([r[2] for r in rows], dtype=np.float64)
        if rows:
            self.matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        else:
            self.matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)

    # --- 書き込み ---
    def add_chunk(
        self, guild_id: int, channel_id: int, channel_name: str, messages: list[discord.Message],
    ) -> int:
        """発言とチャンクを保存する (ベクトル化は後で EmbedWorker が行う). キーワード検索はすぐ使える."""
        cur = self.db.execute(
            "INSERT INTO chunks (guild_id, channel_id, start_at, end_at, embedding) VALUES (?, ?, ?, ?, NULL)",
            (guild_id, channel_id, messages[0].created_at.timestamp(), messages[-1].created_at.timestamp()),
        )
        chunk_id = cur.lastrowid
        self.db.executemany(
            "INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (m.id, guild_id, channel_id, channel_name, m.author.id, m.author.display_name,
                 message_text(m), m.created_at.timestamp(), chunk_id)
                for m in messages
            ],
        )
        self.db.commit()
        return chunk_id

    def pending_chunks(self, limit: int) -> list[tuple[int, str]]:
        """まだベクトル化していないチャンクを新しい順に [(chunk_id, 埋め込み用テキスト)]."""
        rows = self.db.execute(
            "SELECT id FROM chunks WHERE embedding IS NULL ORDER BY end_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for (chunk_id,) in rows:
            msgs = self.chunk_messages(chunk_id)
            if not msgs:
                self.db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
                continue
            lines = [f"#{msgs[0].channel_name}", *(f"{m.author_name}: {m.content}" for m in msgs)]
            result.append((chunk_id, "\n".join(lines)[:4000]))
        self.db.commit()
        return result

    def pending_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]

    def set_embedding(self, chunk_id: int, embedding: np.ndarray) -> None:
        row = self.db.execute("SELECT channel_id, start_at FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return
        self.db.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (embedding.tobytes(), chunk_id))
        self.db.commit()
        self.chunk_ids = np.append(self.chunk_ids, chunk_id)
        self.chunk_channels = np.append(self.chunk_channels, row[0])
        self.chunk_times = np.append(self.chunk_times, row[1])
        self.matrix = np.vstack([self.matrix, embedding[None, :]])

    def purge_channel(self, channel_id: int) -> int:
        """チャンネルの発言・チャンクを索引から消す (スレッドは parent を問わず channel_id 単位)."""
        n = self.db.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ?", (channel_id,)).fetchone()[0]
        self.db.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM chunks WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM channel_state WHERE channel_id = ?", (channel_id,))
        self.db.commit()
        if n:
            self._load_vectors()
        return n

    def delete_message(self, message_id: int) -> None:
        self.db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self.db.commit()

    def update_message(self, message_id: int, content: str) -> None:
        self.db.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
        self.db.commit()

    def get_state(self, channel_id: int) -> tuple[int | None, int | None, bool]:
        row = self.db.execute(
            "SELECT newest_id, oldest_id, backfilled FROM channel_state WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        return (row[0], row[1], bool(row[2])) if row else (None, None, False)

    def set_state(self, channel_id: int, newest_id: int | None, oldest_id: int | None,
                  backfilled: bool, added: int) -> None:
        self.db.execute(
            """INSERT INTO channel_state (channel_id, newest_id, oldest_id, backfilled, indexed_messages)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(channel_id) DO UPDATE SET newest_id = excluded.newest_id,
                 oldest_id = excluded.oldest_id, backfilled = excluded.backfilled,
                 indexed_messages = indexed_messages + excluded.indexed_messages""",
            (channel_id, newest_id, oldest_id, int(backfilled), added),
        )
        self.db.commit()

    # --- 読み出し ---
    def vector_search(
        self, query_vec: np.ndarray, channel_ids: set[int], limit: int,
        since: float | None = None, until: float | None = None,
    ) -> list[tuple[int, float]]:
        """コサイン類似度の高いチャンクを返す [(chunk_id, score)]."""
        if not len(self.chunk_ids):
            return []
        mask = np.isin(self.chunk_channels, list(channel_ids))
        if since is not None:
            mask &= self.chunk_times >= since
        if until is not None:
            mask &= self.chunk_times < until
        idx = np.nonzero(mask)[0]
        if not len(idx):
            return []
        scores = self.matrix[idx] @ query_vec
        # 新しい会話を少しだけ優遇する (類似度がほぼ同じなら新しい方を候補に残す)
        age_days = np.maximum(0, (time.time() - self.chunk_times[idx]) / 86400)
        scores = scores + 0.03 * np.power(0.5, age_days / 60)
        order = np.argsort(-scores)[:limit]
        return [(int(self.chunk_ids[idx[i]]), float(scores[i])) for i in order]

    def keyword_search(
        self, keywords: list[str], channel_ids: set[int], limit: int,
        author_ids: list[int] | None = None, since: float | None = None, until: float | None = None,
    ) -> list[int]:
        """キーワードを含む発言のあるチャンク (新しい順).

        英字のキーワードは単語単位で照合する (「ai」が「mail」「said」に一致しないように)。
        """
        if not keywords and not author_ids:
            return []
        where, params = self._filters(channel_ids, author_ids, since, until)
        if keywords:
            where.append("(" + " OR ".join("content LIKE ?" for _ in keywords) + ")")
            params += [f"%{k}%" for k in keywords]
        rows = self.db.execute(
            f"""SELECT chunk_id, content, created_at FROM messages
                WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT ?""",
            [*params, limit * 20],
        ).fetchall()
        patterns = [keyword_pattern(k) for k in keywords]
        result: list[int] = []
        for chunk_id, content, _ in rows:
            if chunk_id is None or chunk_id in result:
                continue
            if patterns and not any(p.search(content) for p in patterns):
                continue
            result.append(chunk_id)
            if len(result) >= limit:
                break
        return result

    def _filters(self, channel_ids, author_ids, since, until) -> tuple[list[str], list]:
        where = [f"channel_id IN ({','.join('?' * len(channel_ids))})"]
        params: list = list(channel_ids)
        if author_ids:
            where.append(f"author_id IN ({','.join('?' * len(author_ids))})")
            params += author_ids
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        if until is not None:
            where.append("created_at < ?")
            params.append(until)
        return where, params

    def recent_chunks(
        self, channel_ids: set[int], limit: int, author_ids: list[int] | None = None,
        since: float | None = None, until: float | None = None,
    ) -> list[int]:
        """条件に合う最近の会話 (話題の指定がないとき用)."""
        where, params = self._filters(channel_ids, author_ids, since, until)
        rows = self.db.execute(
            f"""SELECT chunk_id, MAX(created_at) AS t FROM messages
                WHERE {' AND '.join(where)} GROUP BY chunk_id ORDER BY t DESC LIMIT ?""",
            [*params, limit],
        ).fetchall()
        return [r[0] for r in rows if r[0] is not None]

    def adjacent_chunks(self, chunk_id: int) -> tuple[int | None, int | None]:
        """同じチャンネルで直前・直後の会話 (チャンク)."""
        row = self.db.execute("SELECT channel_id, start_at FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return None, None
        prev = self.db.execute(
            "SELECT id FROM chunks WHERE channel_id = ? AND start_at < ? ORDER BY start_at DESC LIMIT 1", row
        ).fetchone()
        nxt = self.db.execute(
            "SELECT id FROM chunks WHERE channel_id = ? AND start_at > ? ORDER BY start_at ASC LIMIT 1", row
        ).fetchone()
        return (prev[0] if prev else None), (nxt[0] if nxt else None)

    def find_channels(self, name: str, guild_ids: set[int]) -> set[int]:
        """チャンネル名の部分一致."""
        name = name.strip().lstrip("#").removesuffix("チャンネル").strip()
        if not name:
            return set()
        rows = self.db.execute(
            f"""SELECT DISTINCT channel_id FROM messages
                WHERE guild_id IN ({','.join('?' * len(guild_ids))}) AND channel_name LIKE ?""",
            [*guild_ids, f"%{name}%"],
        ).fetchall()
        return {r[0] for r in rows}

    def chunk_messages(self, chunk_id: int) -> list[StoredMessage]:
        rows = self.db.execute(
            "SELECT * FROM messages WHERE chunk_id = ? ORDER BY created_at", (chunk_id,)
        ).fetchall()
        return [StoredMessage(*r) for r in rows]

    def find_authors(self, name: str, guild_ids: set[int]) -> list[int]:
        """表示名の部分一致で発言者を探す."""
        name = name.strip().removesuffix("さん").removesuffix("くん").removesuffix("氏").strip("@ ")
        if not name:
            return []
        rows = self.db.execute(
            f"""SELECT DISTINCT author_id FROM messages
                WHERE guild_id IN ({','.join('?' * len(guild_ids))}) AND author_name LIKE ?""",
            [*guild_ids, f"%{name}%"],
        ).fetchall()
        return [r[0] for r in rows]

    def author_names(self, guild_id: int, limit: int = 60) -> list[str]:
        """発言の多い順の発言者名 (振り分け時に人名を対応付けるため)."""
        rows = self.db.execute(
            """SELECT author_name, COUNT(*) AS c FROM messages WHERE guild_id = ?
               GROUP BY author_name ORDER BY c DESC LIMIT ?""",
            (guild_id, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> tuple[int, int]:
        """(保存済みの発言数, ベクトル化済みのチャンク数)."""
        msgs = self.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return msgs, len(self.chunk_ids)


class Indexer:
    """発言の取り込み (保存) とベクトル化を行うバックグラウンド処理.

    - 保存: 新着はイベントでリアルタイムに受け取り、会話が一段落したら保存する。
      過去ログと取りこぼし (Bot 停止中の発言など) は定期巡回で補う。保存は API を使わないので速い
    - ベクトル化: 未処理のチャンクを新しい順に、埋め込み API の空き枠を使って処理する
    """

    def __init__(
        self,
        client: discord.Client,
        sarashina: SarashinaClient,
        index: SearchIndex,
        channel_ids: set[int] | None,
        guild_ids: set[int],
        backfill_limit: int = 5000,
        backfill_days: float = 365,
        interval: float = 120.0,
        idle: timedelta = timedelta(minutes=3),
        headroom: int = 20,
        exclude_ids: set[int] | None = None,
    ) -> None:
        self.client = client
        # Bot と会話するチャンネルは取り込まない (Bot への依頼文が検索結果に混ざるため)
        self.exclude_ids = exclude_ids or set()
        self.sarashina = sarashina
        self.index = index
        self.channel_ids = channel_ids  # None ならサーバー内の読める全テキストチャンネル
        self.guild_ids = guild_ids
        self.backfill_limit = backfill_limit
        self.backfill_days = backfill_days  # これより古い発言は遡って取り込まない
        self.interval = interval
        self.idle = idle  # これだけ発言が途切れたら会話が一段落したとみなして保存する
        self.headroom = headroom  # 検索クエリ用に残しておく埋め込み API の枠
        self._live: dict[int, list[discord.Message]] = {}  # チャンネルごとの未保存の新着
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        for channel_id in self.exclude_ids:
            removed = self.index.purge_channel(channel_id)
            if removed:
                log.info("取り込み対象外のチャンネル %s の発言 %d 件を索引から削除しました", channel_id, removed)
        if not self._tasks:
            self._tasks = [
                asyncio.create_task(self._crawl_loop()),
                asyncio.create_task(self._flush_loop()),
                asyncio.create_task(self._embed_loop()),
            ]

    def is_indexed(self, channel) -> bool:
        guild = getattr(channel, "guild", None)
        if guild is None or guild.id not in self.guild_ids:
            return False
        if channel.id in self.exclude_ids or getattr(channel, "parent_id", None) in self.exclude_ids:
            return False
        if self.channel_ids is None:
            return True
        return channel.id in self.channel_ids or getattr(channel, "parent_id", None) in self.channel_ids

    def channels(self) -> list[discord.abc.Messageable]:
        result = []
        for guild_id in self.guild_ids:
            guild = self.client.get_guild(guild_id)
            if not guild:
                continue
            for ch in [*guild.text_channels, *guild.threads]:
                if not self.is_indexed(ch):
                    continue
                perms = ch.permissions_for(guild.me)
                if perms.view_channel and perms.read_message_history:
                    result.append(ch)
        return result

    # --- リアルタイム取り込み ---
    def on_message(self, message: discord.Message) -> None:
        """新着発言を受け取る (会話が一段落したら _flush_loop が保存する)."""
        if message.author.bot or not message_text(message) or not self.is_indexed(message.channel):
            return
        newest_id, _, _ = self.index.get_state(message.channel.id)
        if newest_id is None:
            return  # 初回の取り込み前のチャンネルは巡回に任せる
        self._live.setdefault(message.channel.id, []).append(message)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(20)
            now = datetime.now(timezone.utc)
            for channel_id, buf in list(self._live.items()):
                if not buf:
                    continue
                idle = now - buf[-1].created_at >= self.idle
                full = len(buf) >= CHUNK_MAX_MESSAGES or now - buf[0].created_at >= self.idle * 2
                if idle or full:
                    self._live[channel_id] = []
                    try:
                        self._save(buf[0].channel, buf)
                    except Exception:
                        log.exception("新着の保存に失敗しました (channel %s)", channel_id)

    def _save(self, channel, messages: list[discord.Message]) -> int:
        """発言を会話のまとまりに分けて保存し、チャンネルの取り込み位置を進める."""
        messages = [m for m in messages if not m.author.bot and message_text(m)]
        for chunk in split_chunks(messages):
            self.index.add_chunk(channel.guild.id, channel.id, channel.name, chunk)
        if messages:
            newest_id, oldest_id, backfilled = self.index.get_state(channel.id)
            newest = max(newest_id or 0, messages[-1].id)
            self.index.set_state(channel.id, newest, oldest_id or messages[0].id, backfilled, len(messages))
        return len(messages)

    # --- 定期巡回 (初回取り込み・取りこぼし・過去ログ) ---
    async def _crawl_loop(self) -> None:
        await self.client.wait_until_ready()
        while not self.client.is_closed():
            started = time.monotonic()
            channels = sorted(self.channels(), key=lambda c: c.last_message_id or 0, reverse=True)
            for ch in channels:
                await self._safe(self.catch_up, ch)
            for ch in channels:
                await self._safe(self.backfill, ch)
            msgs, embedded = self.index.stats()
            log.info("巡回完了 (%.0f秒): 発言 %d 件 / ベクトル化済み %d チャンク / 未処理 %d チャンク",
                     time.monotonic() - started, msgs, embedded, self.index.pending_count())
            await asyncio.sleep(self.interval)

    async def _safe(self, func, ch) -> None:
        try:
            await func(ch)
        except discord.HTTPException as e:
            log.warning("チャンネル %s の取り込みに失敗: %s", ch.id, e)
        except Exception:
            log.exception("チャンネル %s の取り込みで予期しないエラー", ch.id)

    def _too_old(self, m: discord.Message) -> bool:
        return m.created_at < datetime.now(timezone.utc) - timedelta(days=self.backfill_days)

    async def catch_up(self, channel) -> None:
        """前回の取り込み位置以降の発言 (初回は直近500件) を取り込む.

        リアルタイム取り込み待ちの発言と、まだ会話が続いている直近の発言は除く。
        """
        newest_id, oldest_id, backfilled = self.index.get_state(channel.id)
        cutoff = discord.Object(discord.utils.time_snowflake(datetime.now(timezone.utc) - self.idle))
        if newest_id is None:
            batch = [m async for m in channel.history(limit=500, before=cutoff)][::-1]
        else:
            batch = [m async for m in channel.history(
                limit=1000, after=discord.Object(newest_id), before=cutoff, oldest_first=True)]
        pending = {m.id for m in self._live.get(channel.id, [])}
        batch = [m for m in batch if m.id not in pending]
        if not batch:
            if newest_id is None:
                self.index.set_state(channel.id, None, None, True, 0)
            return
        recent = [m for m in batch if not self._too_old(m)]
        self._save(channel, recent)
        newest_id, oldest_id, _ = self.index.get_state(channel.id)
        # 初回取り込みで既に1年より前まで届いていれば、過去ログの遡りは不要
        done = backfilled or len(recent) < len(batch)
        self.index.set_state(channel.id, max(newest_id or 0, batch[-1].id),
                             min(oldest_id or batch[0].id, batch[0].id), done, 0)

    async def backfill(self, channel) -> None:
        """過去ログを500件ずつ遡る (backfill_days より古い発言、または backfill_limit 件で終了)."""
        newest_id, oldest_id, backfilled = self.index.get_state(channel.id)
        if backfilled or oldest_id is None:
            return
        batch = [m async for m in channel.history(limit=500, before=discord.Object(oldest_id))][::-1]
        total = self.index.db.execute(
            "SELECT indexed_messages FROM channel_state WHERE channel_id = ?", (channel.id,)
        ).fetchone()
        recent = [m for m in batch if not self._too_old(m)]
        done = (
            len(batch) < 500
            or len(recent) < len(batch)
            or bool(total and total[0] + len(recent) >= self.backfill_limit)
        )
        if recent:
            self._save(channel, recent)
        self.index.set_state(channel.id, newest_id, batch[0].id if batch else oldest_id, done, 0)
        if done:
            log.info("チャンネル #%s の過去ログ取り込みが完了しました", channel.name)

    # --- ベクトル化 ---
    async def _embed_loop(self) -> None:
        """未処理のチャンクを新しい順にベクトル化する (検索用の枠を残して空き枠だけ使う)."""
        await self.client.wait_until_ready()
        limiter = self.sarashina.limiter(EMBED_MODEL)
        while not self.client.is_closed():
            pending = self.index.pending_chunks(20)
            if not pending:
                await asyncio.sleep(10)
                continue
            for chunk_id, text in pending:
                await limiter.wait_for_headroom(self.headroom)
                try:
                    vec = await self.sarashina.embed(text, "RETRIEVAL_DOCUMENT")
                    self.index.set_embedding(chunk_id, normalize(vec))
                except Exception:
                    log.exception("ベクトル化に失敗しました (chunk %s)", chunk_id)
                    await asyncio.sleep(5)
