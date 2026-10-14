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

ROUTER_PROMPT = """あなたは Discord Bot の振り分け担当です。ユーザーの最新の投稿が「過去の会話の検索」かどうかを判定し、JSON だけを返してください。

検索 (mode=search) の例:
- 「〇〇の話どこだっけ？」「〇〇について誰か言ってなかった？」→ type=locate (発言の場所を探す)
- 「〇〇と議論していた内容をまとめて」「〇〇さんの最近の発言をまとめて」→ type=summarize (過去の会話をまとめる)
- 直前の検索結果への追加の依頼 (「古すぎる」「もっと最近のを」「他には？」「直近の会話を持ってきて」) も search。
  このときは直前の検索の条件 (話題・人・チャンネル・type) を引き継ぎ、必要に応じて order 等を変える
それ以外 (一般的な質問・雑談・依頼・Bot の回答への感想や指摘) は mode=chat。

今日は {today} です。
質問者: {asker}
このサーバーのチャンネル: {channels}
発言者の名前の一覧: {authors}

出力形式:
{{"mode": "search" または "chat",
  "type": "locate" または "summarize",
  "query": "検索したい話題を簡潔に。人名・チャンネル名・『どこだっけ』『まとめて』等は含めない。話題の指定が無ければ空文字",
  "keywords": ["本文に含まれていそうな語", ...],
  "person": "特定の人の発言・その人との会話なら、発言者の名前の一覧の中から最も当てはまる名前 (無ければ null)",
  "channel": "特定のチャンネルの話なら、チャンネル一覧の中の名前 (無ければ null)",
  "order": "recent" または "relevance",
  "since": "YYYY-MM-DD または null",
  "until": "YYYY-MM-DD または null (この日を含まない)"}}

注意:
- keywords は話題そのものを表す語を2〜5個 (表記ゆれ・言い換え・英語表記も含める)。人名や「話」「議論」「内容」「まとめ」「どこ」などの一般的な語は入れない
- person は一覧の表記で返す。ローマ字・ニックネーム・苗字の読みなども考慮して対応付ける (例: 苗字「山田」→ 一覧の「yamada_x」)。「自分」「俺」なら質問者
- order は「最近」「直近」「最新」「古すぎる」など新しさを求めていれば "recent"、それ以外は "relevance"
- since/until は「先週」「10月」など期間が明示されている場合だけ設定し、それ以外は必ず null
- 例: 「お知らせで告知してた内容をまとめて」→ channel="お知らせ・告知" (一覧にあるもの), query=""
- 例: 「Aさんとキャッシュの設定について話したのどこ？」→ person="A", query="キャッシュの設定", type=locate
- 例: 「Bの発言をまとめて」→ person=(一覧のBに当たる名前), query="", keywords=[], type=summarize"""

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


# 検索の依頼らしさの手がかり. これらを含まない投稿は振り分け (nano) を呼ばずに普通の質問として扱う
# (Bot の回答への感想・指摘が誤って検索扱いになるのを防ぐ)
SEARCH_CUES = re.compile(
    r"どこ|探し|検索|まとめ|要約|言って|話して|話した|話して|議論|発言|言動|ログ|履歴|いつ|誰が|だれが"
    r"|書いて(た|い)|話題|やりとり|やり取り|会話"
)
# 直前の検索結果への追加の依頼の手がかり
FOLLOWUP_CUES = re.compile(r"古|最近|直近|最新|新し|他に|ほかに|もっと|続き|別の|それ以外|全部|詳しく")
# 新しさを求める手がかり
RECENT_CUES = re.compile(r"最近|直近|最新|新し|古すぎ|古い|今日|昨日|さっき|今週")
# 「〇〇の発言をまとめて」のように、話題ではなく人やチャンネルだけを指す語
GENERIC_TOPIC = re.compile(r"の?(言動|発言|会話|話|内容|議論|やりとり|やり取り|書き込み|投稿|様子|最近の|直近の)$|^(最近|直近)の?")

# 検索結果の並び順で「新しさ」をどれだけ重視するか (0〜1)
RECENCY_WEIGHT = 0.4


@dataclass
class SearchRequest:
    type: str  # locate / summarize
    query: str
    keywords: list[str] = field(default_factory=list)
    person: str | None = None
    channel: str | None = None
    order: str = "relevance"  # relevance / recent
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

    async def route(
        self, text: str, channel_names: list[str] | None = None, asker: str = "",
        recent: list[dict] | None = None, guild_id: int | None = None,
    ) -> SearchRequest | None:
        """検索なら SearchRequest、普通の質問なら None.

        recent: 直前の会話 (直前の検索結果への追加の依頼を解釈するため)
        """
        forced = False
        for prefix in ("検索:", "検索：", "🔍"):
            if text.startswith(prefix):
                text, forced = text[len(prefix):].strip(), True
        last_bot = next((t["content"] for t in reversed(recent or []) if t["role"] == "assistant"), "")
        followup = last_bot.startswith(("🔍", "📝")) and bool(FOLLOWUP_CUES.search(text))
        if not forced and not followup and not SEARCH_CUES.search(text):
            return None
        raw = await self.sarashina.chat(
            [
                {"role": "system", "content": ROUTER_PROMPT.format(
                    today=datetime.now().strftime("%Y-%m-%d (%a)"),
                    channels=", ".join(channel_names or []) or "(不明)",
                    asker=asker or "(不明)",
                    authors=", ".join(self.index.author_names(guild_id) if guild_id else []) or "(不明)",
                )},
                {"role": "user", "content": self._router_input(text, recent)},
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
        # 「山下の言動」のように人名＋一般語だけのクエリは、話題なしとして扱う
        if person or channel:
            names = [n for n in (person, channel) if n]
            # 元の表記 (例: 苗字) が keywords に入っていればそれも人名として除く
            aliases = [k for k in keywords if k in text and not any(k in n for n in names) and len(k) <= 6
                       and re.search(rf"{re.escape(k)}(さん|くん|氏)?(の|と|が)", text)]
            stripped = query
            for n in [*names, *aliases]:
                stripped = stripped.replace(n, "")
            stripped = GENERIC_TOPIC.sub("", stripped.strip()).strip(" のとをが")
            # 残りが keywords にある短い和語1語だけなら、人名の別表記 (例: 苗字) とみなす
            if stripped in keywords and re.fullmatch(r"[\u3040-\u30ff\u4e00-\u9fff]{1,4}", stripped):
                aliases.append(stripped)
                stripped = ""
            if not stripped:
                query = ""
                keywords = [k for k in keywords if k not in aliases]
        # 新しさを求めていない新規の検索では、関連度順にする (直前の依頼の「最近」を引き継がない)
        order = "recent" if data.get("order") == "recent" else "relevance"
        if order == "recent" and not RECENT_CUES.search(text) and not followup:
            order = "relevance"
        until = _parse_date(data.get("until"))
        if order == "recent" and not re.search(r"\d+月|\d+日|先週|先月|昨年|去年", text):
            until = None  # 「古すぎる」等で終了日を勝手に設定しない
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
            order=order,
            since=_parse_date(data.get("since")),
            until=until,
        )

    @staticmethod
    def _router_input(text: str, recent: list[dict] | None) -> str:
        if not recent:
            return text
        lines = []
        for t in recent[-6:]:
            who = "Bot" if t["role"] == "assistant" else "ユーザー"
            lines.append(f"{who}: {t['content'][:300]}")
        return "# 直前の会話\n" + "\n".join(lines) + f"\n\n# 判定する最新の投稿\n{text}"

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
            if req.person not in req.query:
                req.query = f"{req.person} {req.query}".strip()
            if req.person not in req.keywords:
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

        # 会話 (チャンク) 単位で rerank. 「最近の」指定なら新しい候補を優先して rerank に回す
        if req.order == "recent":
            hits.sort(key=lambda h: -h.messages[-1].created_at)
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
            top = list(hits)  # 話題なしの指定は、条件に合う会話をそのまま使う
        top = self._order(req, top)[:limit]

        # 関連判定でリンク先が決まらなかった会話は、キーワード/rerank で発言を特定する
        await asyncio.gather(*(
            self._pick_best(rerank_query, h, author_ids, req.keywords) for h in top if h.best is None
        ))
        return top

    @staticmethod
    def _order(req: SearchRequest, hits: list[Hit]) -> list[Hit]:
        """関連度の順位と新しさを組み合わせて並べ替える (新しい会話ほど上に来やすい).

        「最近の」指定や話題なしの検索では、単純に新しい順。
        """
        if req.order == "recent" or not req.query.strip():
            return sorted(hits, key=lambda h: -h.messages[-1].created_at)
        # 新しさは結果の中での相対順位で評価する (会話がまばらなサーバーでも効くように)
        n = len(hits)
        newest_first = sorted(hits, key=lambda h: -h.messages[-1].created_at)
        recency = {id(h): 1 - i / max(n, 1) for i, h in enumerate(newest_first)}

        def score(item: tuple[int, Hit]) -> float:
            rank, h = item
            relevance = 1 - rank / max(n, 1)
            return (1 - RECENCY_WEIGHT) * relevance + RECENCY_WEIGHT * recency[id(h)]

        return [h for _, h in sorted(enumerate(hits), key=score, reverse=True)]

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
