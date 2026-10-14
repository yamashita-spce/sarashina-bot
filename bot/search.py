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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import discord

from sarashina import SarashinaClient
from search_index import SearchIndex, StoredMessage, keyword_pattern, normalize

log = logging.getLogger("sarashina-bot.search")

ROUTER_PROMPT = """あなたは Discord Bot の振り分け担当です。ユーザーの最新の投稿が「過去の会話の検索」かどうかを判定し、JSON だけを返してください。

検索 (mode=search) の例:
- 「〇〇の話どこだっけ？」「〇〇について誰か言ってなかった？」→ type=locate (発言の場所を探す)
- 「〇〇と議論していた内容をまとめて」「〇〇さんの最近の発言をまとめて」→ type=summarize (過去の会話をまとめる)
- 直前の検索結果への追加の依頼 (「古すぎる」「もっと最近のを」「他には？」「直近の会話を持ってきて」) も search。
- Bot への感想・評価・からかい (「お前には検索は早かったか」「使えないな」) や、直前の回答の書き直し依頼 (「文章で出力して」) は chat。
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

NEXT_QUERY_PROMPT = """あなたはサーバーの過去ログを調べる検索担当です。ユーザーの依頼、これまでに試した検索語、
これまでに見つかった会話の抜粋 (関連が高そうな順) が与えられます。

1. 見つかった会話で依頼に十分答えられるかを判断する
2. 足りなければ、まだ見つかっていない情報を見つけるための次の検索を最大2つ提案する
   - 言い換え・類義語・関連する固有名詞 (店名・製品名・サービス名など)・英語/カタカナ表記・略称
   - 見つかった会話に出てきた手がかり (関連する話題・名前) を使って広げる
   - 既に試した検索語と同じ・ほぼ同じものは出さない
{extra}
JSON だけを返す:
{{"sufficient": true または false, "missing": "まだ足りない情報 (十分なら空文字)",
 "next": [{{"query": "次の検索語 (短く)", "keywords": ["本文に含まれていそうな語", ...]}}, ...]}}"""

KNOWLEDGE_PROMPT = """あなたは判定係です。ユーザーの質問と、サーバーの過去の会話の候補が与えられます。
各候補は [候補番号] の見出しの後に、発言が (発言番号) 付きで並んでいます。
質問に答えるのに実際に役立つ具体的な情報 (事実・決定事項・経験・意見) が発言の本文に書かれている候補だけを選び、
(URL だけの発言はリンク先の内容が分からないので根拠にならない)
その中で最も役立つ発言の番号を1つ選んでください。話題が少し似ているだけの候補は選ばないこと。
挨拶・雑談・一般常識で答えられる質問なら、空配列を返すこと。
出力形式: {"results": [{"candidate": 候補番号, "message": 発言番号}, ...]}"""

SUMMARIZE_PROMPT = """あなたは Discord サーバーの過去ログを調べるアシスタントです。
以下は検索で見つかった過去の会話です。各発言の先頭に [番号] が付いています。
これらの発言だけを根拠に、ユーザーの依頼に日本語で簡潔に答えてください。

ルール:
- 根拠にした発言は、文末などに [番号] の形で引用する (例: 〜で決まった [3][5])
- 会話に書かれていないことは推測で補わない
- 「見つかりませんでした」「答えられません」で終わらせない。見つかった会話から分かることを中心に答え、足りない点があれば最後に一言だけ添える
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
    before: list[StoredMessage] = field(default_factory=list)  # 前の会話 (まとめる時の文脈用)
    after: list[StoredMessage] = field(default_factory=list)  # 後の会話
    approx: bool = False  # 完全には一致しないが近い話題として出す結果


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
        default_days: float = 365, max_per_author: int = 2,
        min_rounds: int = 2, max_rounds: int = 4,
    ) -> None:
        self.sarashina = sarashina
        self.index = index
        self.router_model = router_model
        self.results = results
        self.default_days = default_days  # 期間指定が無いときに探す範囲 (日)
        self.max_per_author = max_per_author  # 結果に同じ人の発言が並びすぎないように
        self.min_rounds = min_rounds  # 検索を最低何往復するか
        self.max_rounds = max_rounds  # 検索を最大何往復するか

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

    async def search(
        self, req: SearchRequest, channel_ids: set[int], guild_id: int, limit: int,
        question: str = "", progress=None, deadline: float | None = None,
    ) -> list[Hit]:
        """候補を集めて並べ替える.

        期間の指定が無ければ直近 default_days 日を探し、見つからなければ全期間で探し直す。
        期間指定で何も見つからない場合も、期間なしで探し直す。
        """
        explicit = req.since is not None or req.until is not None
        if not explicit and self.default_days:
            req.since = datetime.now().timestamp() - self.default_days * 86400
        hits = await self._search(req, channel_ids, guild_id, limit, question, progress, deadline)
        if not hits and (req.since or req.until):
            log.info("期間内で見つからなかったため、全期間で再検索します")
            if progress:
                await progress("期間を広げて探し直しています", "直近1年では見つからなかったので全期間を探します")
            req.since = req.until = None
            hits = await self._search(req, channel_ids, guild_id, limit, question, progress, deadline)
        return hits

    async def _search(
        self, req: SearchRequest, channel_ids: set[int], guild_id: int, limit: int,
        question: str = "", progress=None, deadline: float | None = None,
    ) -> list[Hit]:
        """何往復かして候補を集め、rerank → 関連判定で絞り込む.

        1回目は依頼の話題で探し、2回目以降は nano が「足りない情報」と次の検索語 (言い換え・関連語・
        見つかった会話に出てきた手がかり) を考えて探し直す。最低 min_rounds 回、最大 max_rounds 回。
        """
        async def report(label: str, note: str | None = None) -> None:
            if progress:
                await progress(label, note)

        all_channels = set(channel_ids)
        author_ids, channel_ids = self._resolve(req, channel_ids, guild_id)
        topic = req.query.strip()
        rerank_query = topic or req.person or req.channel or ""

        if not topic:
            # 話題の指定が無い (「〇〇さんの発言」「#お知らせの内容」等) 場合は、条件に合う最近の会話
            await report("条件に合う最近の会話を集めています")
            pool = self._to_hits(
                self.index.recent_chunks(channel_ids, 20, author_ids, req.since, req.until), author_ids)
            top = sorted(pool, key=lambda h: -h.messages[-1].created_at)[:limit]
        else:
            top = await self._deep_search(
                req, topic, rerank_query, channel_ids, all_channels, author_ids, limit, question,
                report, deadline,
            )
            if top is None:
                return []

        # 関連判定でリンク先が決まらなかった会話は、キーワード/rerank で発言を特定する
        await asyncio.gather(*(
            self._pick_best(rerank_query, h, author_ids, req.keywords) for h in top if h.best is None
        ))
        # まとめる場合は、前後の会話も読んで話の流れが途中で切れないようにする
        if req.type == "summarize" and top:
            await report("前後の会話も読んでいます")
            for h in top:
                prev_id, next_id = self.index.adjacent_chunks(h.chunk_id)
                h.before = self.index.chunk_messages(prev_id)[-4:] if prev_id else []
                h.after = self.index.chunk_messages(next_id)[:4] if next_id else []
        return top

    async def _deep_search(
        self, req: SearchRequest, topic: str, rerank_query: str, channel_ids: set[int],
        all_channels: set[int], author_ids, limit: int, question: str, report, deadline: float | None,
    ) -> list[Hit] | None:
        """十分な情報が集まるまで何往復も検索する.

        - 2回目以降は nano が「足りない情報」と次の検索語 (言い換え・関連語・手がかり) を考える
        - 言い換えで新しい会話が見つからなくなったら、条件を広げる (期間 → チャンネル → 人)
        - deadline (mini の回答枠が空く時刻) までは、十分でも検索を続けて情報を厚くする
        - 最後まで一致する会話が無くても、近い話題の会話を「近い話題」として返す (諦めない)
        """
        pool: dict[int, Hit] = {}
        tried: list[str] = []
        queries = [(topic, list(req.keywords))]
        ranked: list[Hit] = []
        broaden = [k for k, cond in (
            ("period", req.since is not None or req.until is not None),
            ("channel", channel_ids != all_channels),
            ("person", bool(author_ids)),
        ) if cond]
        rnd, stale = 0, 0
        while rnd < self.max_rounds:
            rnd += 1
            before_ids = set(pool)
            for q, kws in queries:
                tried.append(q)
                shown = "」「".join(dict.fromkeys([q, *kws[:2]]))
                await report(f"{rnd}回目: 「{shown}」で探しています")
                for h in await self._gather(q, kws, channel_ids, author_ids, req):
                    pool.setdefault(h.chunk_id, h)
            if pool:
                ranked = await self._rank(rerank_query, list(pool.values()))
                pool = {h.chunk_id: h for h in ranked[:24]}  # 元の依頼との関連が高い24件に絞る
            found = len(set(pool) - before_ids)
            stale = 0 if found else stale + 1
            await report(
                f"{rnd}回目の結果を確かめています",
                f"{rnd}回目で関連しそうな会話が新たに{found}件見つかりました" if found
                else f"{rnd}回目では新しい会話は見つかりませんでした",
            )
            if rnd >= self.max_rounds:
                break
            time_left = deadline is not None and deadline - time.monotonic() > 4
            decision = await self._next_queries(
                question or topic, tried, ranked[:8], must_continue=rnd < self.min_rounds or time_left)
            nexts = [
                (n["query"].strip(), [k for k in n.get("keywords") or [] if isinstance(k, str)])
                for n in decision.get("next") or []
                if isinstance(n, dict) and isinstance(n.get("query"), str) and n["query"].strip()
                and n["query"].strip() not in tried
            ][:2]
            sufficient = bool(decision.get("sufficient")) and bool(pool)
            log.info("検索 %d 回目: %d 件 (新規 %d) / 十分=%s / 次: %s", rnd, len(pool), found,
                     sufficient, [q for q, _ in nexts])
            if sufficient and rnd >= self.min_rounds and not time_left:
                break
            # 言い換えでも新しい会話が見つからない → 条件を広げて探し直す
            if (stale >= 2 or not nexts or not pool) and broaden:
                step = broaden.pop(0)
                if step == "period":
                    req.since = req.until = None
                    await report("期間を広げて探し直しています", "期間を限定せずに探します")
                elif step == "channel":
                    channel_ids = set(all_channels)
                    await report("ほかのチャンネルも探しています", "チャンネルを限定せずに探します")
                elif step == "person":
                    author_ids = None
                    await report("ほかの人の発言も探しています", "発言者を限定せずに探します")
                queries = [(topic, list(req.keywords)), *nexts[:1]]
                stale = 0
                continue
            if not nexts:
                break
            if decision.get("missing") and not sufficient:
                await report("足りない情報を探しに行きます", f"不足: {str(decision['missing'])[:40]}")
            elif sufficient:
                await report("待ち時間を使って、さらに関連する話を探しています")
            queries = nexts
        if not pool:
            return None
        await report(f"見つかった{len(pool)}件の会話を読んで絞り込んでいます")
        top = await self._filter_relevant(req, ranked[:10])
        if not top:
            # 完全に一致する会話が無くても、関連が高い会話を「近い話題」として出す
            top = ranked[:3]
            for h in top:
                h.approx = True
        return self._diversify(req, self._order(req, top), author_ids)[:limit]

    def _resolve(self, req: SearchRequest, channel_ids: set[int], guild_id: int):
        """人名・チャンネル名を ID に解決する. 見つからなければ話題の一部として扱う."""
        author_ids = self.index.find_authors(req.person, {guild_id}) if req.person else None
        if req.person and not author_ids:
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
                req.query = f"{req.channel} {req.query}".strip()
                req.keywords.append(req.channel)
        return author_ids, channel_ids

    def _to_hits(self, chunk_ids: list[int], author_ids) -> list[Hit]:
        hits, seen = [], set()
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            msgs = self.index.chunk_messages(cid)
            if not msgs or (author_ids and not any(m.author_id in author_ids for m in msgs)):
                continue
            hits.append(Hit(cid, 0.0, msgs))
        return hits

    async def _gather(self, query: str, keywords: list[str], channel_ids, author_ids, req) -> list[Hit]:
        """1つの検索語で、意味の近い会話 (embedding) とキーワードを含む会話を集める."""
        qvec = normalize(await self.sarashina.embed(query, "RETRIEVAL_QUERY"))
        ids = [cid for cid, _ in self.index.vector_search(qvec, channel_ids, 15, req.since, req.until)]
        ids += self.index.keyword_search(keywords or [query], channel_ids, 15, author_ids, req.since, req.until)
        hits = self._to_hits(ids, author_ids)
        if req.order == "recent":
            hits.sort(key=lambda h: -h.messages[-1].created_at)
        return hits

    async def _rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """rerank (20件ずつ) で元の依頼との関連度順に並べる."""
        for k in range(0, len(hits), 20):
            batch = hits[k:k + 20]
            for i, score in await self.sarashina.rerank(query, [self._chunk_text(h.messages) for h in batch]):
                batch[i].score = score
        return sorted(hits, key=lambda h: -h.score)

    async def _next_queries(
        self, question: str, tried: list[str], top: list[Hit], must_continue: bool = False
    ) -> dict:
        """見つかった会話で足りるかを nano に判定させ、足りなければ次の検索語を出させる."""
        listing = "\n\n".join(
            f"[{n}] #{h.messages[0].channel_name} {h.messages[0].when}\n{self._chunk_text(h.messages)[:400]}"
            for n, h in enumerate(top)
        ) or "(まだ見つかっていない)"
        try:
            raw = await self.sarashina.chat(
                [
                    {"role": "system", "content": NEXT_QUERY_PROMPT.format(extra=(
                        "※ まだ検索が足りないので、十分かどうかに関わらず、次の検索を必ず1つ以上提案すること。"
                        if must_continue else ""))},
                    {"role": "user", "content": f"# 依頼\n{question}\n\n# 試した検索語\n{', '.join(tried)}\n\n# 見つかった会話\n{listing}"},
                ],
                model=self.router_model, max_completion_tokens=400, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(raw)
        except Exception:
            log.warning("次の検索語の生成に失敗しました", exc_info=True)
            return {"sufficient": True, "next": []}

    def _diversify(self, req: SearchRequest, hits: list[Hit], author_ids) -> list[Hit]:
        """同じ人の発言ばかりにならないよう、1人あたり max_per_author 件までにする.

        人を指定した検索では制限しない。制限で足りなくなる場合は、後ろに回した結果で埋める。
        """
        if author_ids or self.max_per_author <= 0:
            return hits
        counts: dict[int, int] = {}
        picked, deferred = [], []
        for h in hits:
            m = h.best or max(h.messages, key=lambda x: len(x.content))
            if counts.get(m.author_id, 0) >= self.max_per_author:
                deferred.append(h)
                continue
            counts[m.author_id] = counts.get(m.author_id, 0) + 1
            picked.append(h)
        return picked + deferred

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

    async def related(
        self, question: str, channel_ids: set[int], exclude_after: float | None = None,
        exclude_channel: int | None = None, limit: int = 3,
    ) -> list[Hit]:
        """普通の質問に答えるための RAG: 質問に役立つ過去の会話を探す.

        embedding で候補を集め → rerank で並べ替え → nano で「本当に役立つか」を判定。
        質問しているチャンネルの直近の会話 (会話履歴として別に渡す分) は除く。
        """
        if len(question) < 5 or not len(self.index.chunk_ids):
            return []
        qvec = normalize(await self.sarashina.embed(question, "RETRIEVAL_QUERY"))
        hits = []
        for cid, _ in self.index.vector_search(qvec, channel_ids, 15):
            msgs = self.index.chunk_messages(cid)
            if not msgs:
                continue
            if exclude_channel and exclude_after and msgs[0].channel_id == exclude_channel \
                    and msgs[-1].created_at >= exclude_after:
                continue
            hits.append(Hit(cid, 0.0, msgs))
        if not hits:
            return []
        ranked = await self.sarashina.rerank(question, [self._chunk_text(h.messages) for h in hits])
        ordered = [hits[i] for i, _ in ranked[:6]]
        picked = await self._filter_relevant(
            SearchRequest("locate", question), ordered, prompt=KNOWLEDGE_PROMPT, label="ユーザーの質問",
            keep_phrase=False,
        )
        return picked[:limit]

    @staticmethod
    def knowledge_block(hits: list[Hit]) -> str:
        """回答モデルに渡す「過去の関連する会話」の参考情報. 各会話に [参N] の番号を付ける."""
        blocks = []
        for n, h in enumerate(hits, 1):
            first = h.messages[0]
            lines = [f"[参{n}] #{first.channel_name} {first.when}"]
            lines += [f"{m.author_name}: {m.content[:300]}" for m in h.messages]
            blocks.append("\n".join(lines)[:1500])
        return (
            "## 参考: このサーバーの過去の会話から見つかった関連情報\n"
            "質問に答えるのに役立つ場合だけ使い、使った箇所には [参1] のように番号を付けること。"
            "役立たなければ無視し、番号も付けないこと。\n\n" + "\n\n".join(blocks)
        )

    @staticmethod
    def link_knowledge(answer: str, hits: list[Hit]) -> str:
        """回答中の [参N] を発言へのリンクにし、参照した会話を末尾に添える."""
        used = set()

        def link(match: re.Match) -> str:
            n = int(match.group(1))
            if 1 <= n <= len(hits):
                used.add(n)
                m = hits[n - 1].best or hits[n - 1].messages[0]
                return f"[[参{n}]]({m.jump_url})"
            return ""

        answer = re.sub(r"\[参(\d)\](?!\()", link, answer)
        if not used:
            return answer
        refs = []
        for n in sorted(used):
            m = hits[n - 1].best or hits[n - 1].messages[0]
            refs.append(f"[参{n}] <#{m.channel_id}> {m.when} {m.author_name} [→ 移動]({m.jump_url})")
        return answer + "\n\n-# 参考にした過去の会話\n" + "\n".join(f"-# {r}" for r in refs)

    async def _filter_relevant(
        self, req: SearchRequest, hits: list[Hit], prompt: str = RELEVANCE_PROMPT,
        label: str = "探している話題", keep_phrase: bool = True,
    ) -> list[Hit]:
        """nano に会話を読ませ、関連する会話と、その中心となる発言 (リンク先) を選ばせる."""
        blocks = []
        for n, h in enumerate(hits):
            lines = [f"[{n}]"]
            for k, m in enumerate(h.messages):
                lines.append(f"({k}) {m.author_name}: {m.content[:200]}")
            blocks.append("\n".join(lines)[:1500])
        raw = await self.sarashina.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"# {label}\n{req.query}\n\n# 候補\n" + "\n\n".join(blocks)},
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
        # 探している語句そのものを含む候補は、判定で落ちても残す (取りこぼし防止)
        # ※ 個々のキーワード (「検索」など) の部分一致では残さない. 無関係な会話が混ざるため
        phrase = req.query.strip()
        if keep_phrase and len(phrase) >= 2:
            pattern = keyword_pattern(phrase)
            for h in hits:
                if h not in result and any(pattern.search(m.content) for m in h.messages):
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
        if hits and all(h.approx for h in hits):
            lines = [f"🔍 **「{self._label(req)}」** にぴったり当てはまる発言は無かったので、近い話題の発言を挙げます"]
        else:
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

    async def summarize(
        self, req: SearchRequest, hits: list[Hit], user_text: str, slot: float | None = None
    ) -> str:
        """見つかった会話を mini でまとめ、[番号] を発言へのリンクに置き換える."""
        hits = sorted(hits, key=lambda h: h.messages[0].created_at)
        numbered: list[StoredMessage] = []
        blocks = []
        for h in hits:
            first = h.messages[0]
            lines = [f"--- <#{first.channel_id}> {first.when} 〜 ---"]
            for m in [*h.before, *h.messages, *h.after]:
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
            slot=slot,
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
