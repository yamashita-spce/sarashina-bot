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


def chunk_document(channel_name: str, messages: list[discord.Message]) -> str:
    """埋め込み用のテキスト."""
    lines = [f"#{channel_name}"]
    for m in messages:
        lines.append(f"{m.author.display_name}: {message_text(m)}")
    return "\n".join(lines)[:4000]


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
        self, guild_id: int, channel_id: int, channel_name: str,
        messages: list[discord.Message], embedding: np.ndarray,
    ) -> None:
        cur = self.db.execute(
            "INSERT INTO chunks (guild_id, channel_id, start_at, end_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (guild_id, channel_id, messages[0].created_at.timestamp(),
             messages[-1].created_at.timestamp(), embedding.tobytes()),
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
        self.chunk_ids = np.append(self.chunk_ids, chunk_id)
        self.chunk_channels = np.append(self.chunk_channels, channel_id)
        self.chunk_times = np.append(self.chunk_times, messages[0].created_at.timestamp())
        self.matrix = np.vstack([self.matrix, embedding[None, :]])

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
        """キーワードを含む発言のあるチャンク (新しい順)."""
        if not keywords and not author_ids:
            return []
        where, params = self._filters(channel_ids, author_ids, since, until)
        if keywords:
            where.append("(" + " OR ".join("content LIKE ?" for _ in keywords) + ")")
            params += [f"%{k}%" for k in keywords]
        rows = self.db.execute(
            f"""SELECT chunk_id, MAX(created_at) AS t FROM messages
                WHERE {' AND '.join(where)} GROUP BY chunk_id ORDER BY t DESC LIMIT ?""",
            [*params, limit],
        ).fetchall()
        return [r[0] for r in rows if r[0] is not None]

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
        msgs = self.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return msgs, len(self.chunk_ids)


class Indexer:
    """対象チャンネルの過去ログと新着を、埋め込み API の空き枠を使って少しずつ索引付けする."""

    def __init__(
        self,
        client: discord.Client,
        sarashina: SarashinaClient,
        index: SearchIndex,
        channel_ids: set[int] | None,
        guild_ids: set[int],
        backfill_limit: int = 5000,
        interval: float = 120.0,
        settle: timedelta = timedelta(minutes=5),
        headroom: int = 20,
    ) -> None:
        self.client = client
        self.sarashina = sarashina
        self.index = index
        self.channel_ids = channel_ids  # None ならサーバー内の読める全テキストチャンネル
        self.guild_ids = guild_ids
        self.backfill_limit = backfill_limit
        self.interval = interval
        self.settle = settle  # 会話が落ち着いてから索引付けする (チャンクが途中で切れないように)
        self.headroom = headroom  # 検索クエリ用に残しておく埋め込み API の枠
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._loop())

    def channels(self) -> list[discord.abc.Messageable]:
        result = []
        for guild_id in self.guild_ids:
            guild = self.client.get_guild(guild_id)
            if not guild:
                continue
            for ch in [*guild.text_channels, *guild.threads]:
                if self.channel_ids is not None and ch.id not in self.channel_ids and getattr(ch, "parent_id", None) not in self.channel_ids:
                    continue
                perms = ch.permissions_for(guild.me)
                if perms.view_channel and perms.read_message_history:
                    result.append(ch)
        return result

    async def _loop(self) -> None:
        await self.client.wait_until_ready()
        while not self.client.is_closed():
            started = time.monotonic()
            for ch in self.channels():
                try:
                    await self.index_channel(ch)
                except discord.HTTPException as e:
                    log.warning("チャンネル %s の索引付けに失敗: %s", ch.id, e)
                except Exception:
                    log.exception("チャンネル %s の索引付けで予期しないエラー", ch.id)
            msgs, chunks = self.index.stats()
            log.info("索引付けサイクル完了 (%.0f秒): 発言 %d 件 / チャンク %d 件",
                     time.monotonic() - started, msgs, chunks)
            await asyncio.sleep(self.interval)

    async def index_channel(self, channel) -> None:
        newest_id, oldest_id, backfilled = self.index.get_state(channel.id)
        cutoff = discord.Object(discord.utils.time_snowflake(datetime.now(timezone.utc) - self.settle))

        # 1. 新着 (前回の最新以降、落ち着いた発言のみ)
        if newest_id is None:
            batch = [m async for m in channel.history(limit=200, before=cutoff)][::-1]
        else:
            batch = [m async for m in channel.history(
                limit=500, after=discord.Object(newest_id), before=cutoff, oldest_first=True)]
        if batch:
            added = await self._store(channel, batch)
            newest_id = batch[-1].id
            oldest_id = oldest_id or batch[0].id
            self.index.set_state(channel.id, newest_id, oldest_id, backfilled, added)

        # 2. 過去ログ (少しずつ遡る)
        if not backfilled and oldest_id is not None:
            before = discord.Object(oldest_id)
            batch = [m async for m in channel.history(limit=200, before=before)][::-1]
            total = self.index.db.execute(
                "SELECT indexed_messages FROM channel_state WHERE channel_id = ?", (channel.id,)
            ).fetchone()
            done = len(batch) < 200 or (total and total[0] + len(batch) >= self.backfill_limit)
            added = await self._store(channel, batch) if batch else 0
            if batch:
                oldest_id = batch[0].id
            self.index.set_state(channel.id, newest_id, oldest_id, bool(done), added)
            if done:
                log.info("チャンネル #%s の過去ログ索引付けが完了しました", channel.name)
        elif newest_id is None:
            self.index.set_state(channel.id, None, None, True, 0)

    async def _store(self, channel, messages: list[discord.Message]) -> int:
        """Bot 以外の発言をチャンクに分けて埋め込み、保存する."""
        messages = [m for m in messages if not m.author.bot and message_text(m)]
        added = 0
        for chunk in split_chunks(messages):
            limiter = self.sarashina.limiter(EMBED_MODEL)
            await limiter.wait_for_headroom(self.headroom)
            vec = await self.sarashina.embed(chunk_document(channel.name, chunk), "RETRIEVAL_DOCUMENT")
            self.index.add_chunk(channel.guild.id, channel.id, channel.name, chunk, normalize(vec))
            added += len(chunk)
        return added
