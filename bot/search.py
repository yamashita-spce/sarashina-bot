"""チャンネル横断検索: 質問の振り分け・候補の収集・並べ替え・リンク付きの回答生成.

使う API:
- sarashina3-nano     … 質問が「検索」か「普通の質問」かの判定と、検索条件 (キーワード・人・期間) の抽出
- sarashina3-embedding … 意味の近い会話 (チャンク) の検索
- sarashina3-rerank    … 候補の会話の並べ替えと、会話の中で一番該当する発言の特定
- sarashina3-mini      … 「まとめて」系の質問で、見つかった会話を要約 (引用リンク付き)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import discord

from sarashina import SarashinaClient
from search_index import SearchIndex, StoredMessage, normalize

log = logging.getLogger("sarashina-bot.search")

ROUTER_PROMPT = """あなたは Discord Bot の振り分け担当です。ユーザーの投稿が「過去の会話の検索」かどうかを判定し、JSON だけを返してください。

検索 (mode=search) の例:
- 「〇〇の話どこだっけ？」「〇〇について誰か言ってなかった？」→ type=locate (発言の場所を探す)
- 「〇〇と議論していた内容をまとめて」「先週の〇〇の話を要約して」→ type=summarize (過去の会話をまとめる)
それ以外 (一般的な質問・雑談・依頼) は mode=chat。

今日は {today} です。
このサーバーのチャンネル: {channels}

出力形式:
{{"mode": "search" または "chat",
  "type": "locate" または "summarize",
  "query": "検索したい話題を簡潔に (人名・チャンネル名や『どこだっけ』等は除く。特定の話題が無ければ空文字)",
  "keywords": ["本文に含まれていそうな語", ...],
  "person": "特定の人との会話なら、その人の名前 (無ければ null)",
  "channel": "特定のチャンネルの話なら、上のチャンネル一覧の中の名前 (無ければ null)",
  "since": "YYYY-MM-DD または null",
  "until": "YYYY-MM-DD または null (この日を含まない)"}}

注意:
- keywords は話題そのものを表す語を2〜5個 (表記ゆれ・言い換え・英語表記も含める)。人名や「話」「議論」「内容」「まとめ」「どこ」などの一般的な語は入れない
- since/until は「先週」「10月」など期間が質問に明示されている場合だけ設定し、それ以外は必ず null
- 例: 「お知らせで告知してた内容をまとめて」→ channel="お知らせ・告知" (一覧にあるもの), query=""
- 例: 「たなかさんとRedisについて話したのどこ？」→ person="たなか", query="Redis", type=locate"""

RELEVANCE_PROMPT = """あなたは検索結果の判定係です。ユーザーが探している話題と、候補の会話が与えられます。
各候補は [候補番号] の見出しの後に、発言が (発言番号) 付きで並んでいます。

1. 話題について実際に話している候補だけを選ぶ (単語が一部似ているだけ・無関係な候補は除く)
2. 選んだ候補ごとに、その話題の中心となっている発言の番号を1つ選ぶ (URL だけ・相づちだけの発言は避ける)
関連が強い順に並べ、JSON だけを返してください。該当がなければ空配列。
出力形式: {"results": [{"candidate": 候補番号, "message": 発言番号}, ...]}"""

SUMMARIZE_PROMPT = """あなたは Discord サーバーの過去ログを調べるアシスタントです。
以下は検索で見つかった過去の会話です。各発言の先頭に [番号] が付いています。
これらの発言だけを根拠に、ユーザーの依頼に日本語で簡潔に答えてください。

ルール:
- 根拠にした発言は、文末などに [番号] の形で引用する (例: 〜で決まった [3][5])
- 会話に書かれていないことは推測で補わない
- 依頼が特定の話題を指定していない場合 (「#〇〇の内容をまとめて」「〇〇さんとの会話をまとめて」等) は、見つかった会話全体の要点を話題ごとにまとめる
- 誰が何を言ったか・何が決まったか・未解決の点が分かるようにまとめる
- 前置きは不要"""


@dataclass
class SearchRequest:
    type: str  # locate / summarize
    query: str
    keywords: list[str] = field(default_factory=list)
    person: str | None = None
    channel: str | None = None
    since: float | None = None
    until: float | None = None


@dataclass
class Hit:
    chunk_id: int
    score: float
    messages: list[StoredMessage]
    best: StoredMessage | None = None


def _parse_date(s) -> float | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").timestamp()
    except ValueError:
        return None


class Searcher:
    def __init__(
        self, sarashina: SarashinaClient, index: SearchIndex,
        router_model: str = "sarashina3-nano", results: int = 5,
    ) -> None:
        self.sarashina = sarashina
        self.index = index
        self.router_model = router_model
        self.results = results

    async def route(self, text: str, channel_names: list[str] | None = None) -> SearchRequest | None:
        """検索なら SearchRequest、普通の質問なら None."""
        forced = False
        for prefix in ("検索:", "検索：", "🔍"):
            if text.startswith(prefix):
                text, forced = text[len(prefix):].strip(), True
        raw = await self.sarashina.chat(
            [
                {"role": "system", "content": ROUTER_PROMPT.format(
                    today=datetime.now().strftime("%Y-%m-%d (%a)"),
                    channels=", ".join(channel_names or []) or "(不明)",
                )},
                {"role": "user", "content": text},
            ],
            model=self.router_model,
            max_completion_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("振り分け結果を JSON として読めませんでした: %s", raw[:200])
            return SearchRequest("locate", text) if forced else None
        if data.get("mode") != "search" and not forced:
            return None
        keywords = [k for k in data.get("keywords") or [] if isinstance(k, str) and k.strip()]
        query = (data.get("query") or "").strip()
        channel = data.get("channel") or None
        person = data.get("person") or None
        if not query and not channel and not person:
            query = text
        if not keywords and query and len(query) <= 20:
            keywords = [query]
        return SearchRequest(
            type="summarize" if data.get("type") == "summarize" else "locate",
            query=query,
            keywords=keywords[:6],
            person=person,
            channel=channel,
            since=_parse_date(data.get("since")),
            until=_parse_date(data.get("until")),
        )

    def allowed_channels(self, member: discord.Member | discord.User, guild: discord.Guild | None) -> set[int]:
        """質問者が閲覧できるチャンネルだけを検索対象にする (見えないチャンネルの内容を漏らさない)."""
        if guild is None or not isinstance(member, discord.Member):
            return set()
        allowed = set()
        for ch in [*guild.text_channels, *guild.threads]:
            perms = ch.permissions_for(member)
            if perms.view_channel and perms.read_message_history:
                allowed.add(ch.id)
        return allowed

    async def search(self, req: SearchRequest, channel_ids: set[int], guild_id: int, limit: int) -> list[Hit]:
        """候補を集めて並べ替える. 期間指定で何も見つからなければ期間なしで探し直す."""
        hits = await self._search(req, channel_ids, guild_id, limit)
        if not hits and (req.since or req.until):
            log.info("期間指定で見つからなかったため、期間なしで再検索します")
            req.since = req.until = None
            hits = await self._search(req, channel_ids, guild_id, limit)
        return hits

    async def _search(self, req: SearchRequest, channel_ids: set[int], guild_id: int, limit: int) -> list[Hit]:
        """埋め込み検索とキーワード検索で候補を集め、rerank → 関連判定で絞り込む."""
        author_ids = self.index.find_authors(req.person, {guild_id}) if req.person else None
        if req.person and not author_ids:
            # 人名として見つからなければ話題の一部として扱う
            req.query = f"{req.person} {req.query}".strip()
            req.keywords.append(req.person)
            author_ids = None
        if req.channel:
            named = self.index.find_channels(req.channel, {guild_id}) & channel_ids
            if named:
                channel_ids = named
            else:
                # チャンネル名として見つからなければ話題の一部として扱う
                req.query = f"{req.channel} {req.query}".strip()
                req.keywords.append(req.channel)

        topic = req.query.strip()
        candidates: list[int] = []
        if topic:
            qvec = normalize(await self.sarashina.embed(topic, "RETRIEVAL_QUERY"))
            candidates += [cid for cid, _ in self.index.vector_search(qvec, channel_ids, 30, req.since, req.until)]
            candidates += self.index.keyword_search(
                req.keywords, channel_ids, 30, author_ids, req.since, req.until)
        else:
            # 話題の指定が無い (「〇〇さんとの会話」「#お知らせの内容」等) 場合は、条件に合う最近の会話
            candidates += self.index.recent_chunks(channel_ids, 20, author_ids, req.since, req.until)

        # 重複を除き、人の指定があればその人が参加している会話に絞る
        hits: list[Hit] = []
        seen = set()
        for cid in candidates:
            if cid in seen:
                continue
            seen.add(cid)
            msgs = self.index.chunk_messages(cid)
            if not msgs:
                continue
            if author_ids and not any(m.author_id in author_ids for m in msgs):
                continue
            hits.append(Hit(cid, 0.0, msgs))
        if not hits:
            return []

        # 会話 (チャンク) 単位で rerank
        hits = hits[:20]
        rerank_query = topic or req.person or req.channel or ""
        docs = [self._chunk_text(h.messages) for h in hits]
        ranked = await self.sarashina.rerank(rerank_query, docs)
        ordered = []
        for i, score in ranked:
            hits[i].score = score
            ordered.append(hits[i])

        # rerank のスコアは無関係な候補でも高く出るため、上位候補を nano で関連あり/なし判定する
        if topic:
            top = await self._filter_relevant(req, ordered[:8])
        else:
            # 話題なしの指定は、条件に合う会話を新しい順にそのまま使う
            top = sorted(hits, key=lambda h: -h.messages[-1].created_at)
        top = top[:limit]

        # 関連判定でリンク先が決まらなかった会話は、キーワード/rerank で発言を特定する
        await asyncio.gather(*(
            self._pick_best(rerank_query, h, author_ids, req.keywords) for h in top if h.best is None
        ))
        return top

    async def _filter_relevant(self, req: SearchRequest, hits: list[Hit]) -> list[Hit]:
        """nano に会話を読ませ、関連する会話と、その中心となる発言 (リンク先) を選ばせる."""
        blocks = []
        for n, h in enumerate(hits):
            lines = [f"[{n}]"]
            for k, m in enumerate(h.messages):
                lines.append(f"({k}) {m.author_name}: {m.content[:200]}")
            blocks.append("\n".join(lines)[:1500])
        raw = await self.sarashina.chat(
            [
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": f"# 探している話題\n{req.query}\n\n# 候補\n" + "\n\n".join(blocks)},
            ],
            model=self.router_model,
            max_completion_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result: list[Hit] = []
        try:
            for item in json.loads(raw).get("results") or []:
                c, k = item.get("candidate"), item.get("message")
                if not isinstance(c, int) or not 0 <= c < len(hits) or hits[c] in result:
                    continue
                h = hits[c]
                if isinstance(k, int) and 0 <= k < len(h.messages):
                    h.best = h.messages[k]
                result.append(h)
        except (json.JSONDecodeError, AttributeError, TypeError):
            log.warning("関連判定の結果を読めませんでした: %s", raw[:200])
            return hits
        # キーワードがそのまま含まれる候補は、判定で落ちても残す (取りこぼし防止)
        kw = [k for k in req.keywords if len(k) >= 2]
        for h in hits:
            if h not in result and kw and any(k.lower() in m.content.lower() for m in h.messages for k in kw):
                result.append(h)
        return result

    @staticmethod
    def _chunk_text(msgs: list[StoredMessage]) -> str:
        return "\n".join(f"{m.author_name}: {m.content}" for m in msgs)[:3000]

    async def _pick_best(self, query: str, hit: Hit, author_ids, keywords: list[str]) -> None:
        """会話の中で一番該当する発言 (リンク先) を選ぶ.

        キーワードを含む発言を優先し、無ければ rerank で選ぶ。URL だけ・極端に短い発言は避ける。
        """
        msgs = hit.messages
        if author_ids:
            msgs = [m for m in msgs if m.author_id in author_ids] or msgs
        meaningful = [m for m in msgs if len(re.sub(r"https?://\S+|\s", "", m.content)) >= 6] or msgs
        kw = [k for k in keywords if k]
        with_kw = [m for m in meaningful if any(k.lower() in m.content.lower() for k in kw)]
        pool = with_kw or meaningful
        if len(pool) == 1 or not query:
            hit.best = pool[0]
            return
        try:
            ranked = await self.sarashina.rerank(query, [f"{m.author_name}: {m.content}" for m in pool[:20]])
            hit.best = pool[ranked[0][0]]
        except Exception:
            log.exception("発言の特定に失敗しました")
            hit.best = pool[0]

    # --- 出力 ---
    def format_locate(self, req: SearchRequest, hits: list[Hit]) -> str:
        lines = [f"🔍 **「{self._label(req)}」** に関する発言が見つかりました"]
        for n, h in enumerate(hits, 1):
            m = h.best or h.messages[0]
            snippet = re.sub(r"\s+", " ", m.content)
            if len(snippet) > 90:
                snippet = snippet[:89] + "…"
            lines.append(
                f"\n**{n}.** <#{m.channel_id}> · {m.when} · **{m.author_name}**\n"
                f"> {snippet}\n"
                f"[→ この発言へ移動]({m.jump_url})"
            )
        return "\n".join(lines)

    async def summarize(self, req: SearchRequest, hits: list[Hit], user_text: str) -> str:
        """見つかった会話を mini でまとめ、[番号] を発言へのリンクに置き換える."""
        hits = sorted(hits, key=lambda h: h.messages[0].created_at)
        numbered: list[StoredMessage] = []
        blocks = []
        for h in hits:
            first = h.messages[0]
            lines = [f"--- <#{first.channel_id}> {first.when} 〜 ---"]
            for m in h.messages:
                numbered.append(m)
                lines.append(f"[{len(numbered)}] {m.author_name}: {m.content}")
            blocks.append("\n".join(lines))
        context = "\n\n".join(blocks)[:12000]
        answer = await self.sarashina.chat(
            [
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": f"# 依頼\n{user_text}\n\n# 検索条件\n{self._label(req)}\n\n# 見つかった会話\n{context}"},
            ],
            temperature=0.3,
        )

        def link(match: re.Match) -> str:
            n = int(match.group(1))
            if 1 <= n <= len(numbered):
                return f"[[{n}]]({numbered[n - 1].jump_url})"
            return match.group(0)

        answer = re.sub(r"\[(\d{1,3})\](?!\()", link, answer)
        sources = [
            f"- <#{h.messages[0].channel_id}> {h.messages[0].when} の会話 [→ 移動]({h.messages[0].jump_url})"
            for h in hits
        ]
        return f"📝 **「{self._label(req)}」** に関する会話のまとめ\n\n{answer}\n\n-# 参照した会話\n" + "\n".join(sources)

    @staticmethod
    def _label(req: SearchRequest) -> str:
        parts = [p for p in (req.query, req.person and f"{req.person}との会話", req.channel and f"#{req.channel}") if p]
        return " / ".join(parts) or "検索"
