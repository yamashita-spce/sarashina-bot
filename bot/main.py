"""対象チャンネルの投稿に Sarashina が回答する Discord Bot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from logging.handlers import TimedRotatingFileHandler

import discord
from dotenv import load_dotenv

from context import ContextManager
from sarashina import SarashinaClient, SarashinaError
from search import ComposeResult, SearchRequest, Searcher
from search_index import Indexer, SearchIndex
from thinking import (
    PLAN_LABELS_PROMPT,
    PLAN_PROMPT,
    REVIEW_PROMPT,
    SPINNER,
    THINKING_SYSTEM_PROMPT,
    Animator,
    ThinkingView,
    ThrottledEditor,
    truncate,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

log = logging.getLogger("sarashina-bot")

DISCORD_LIMIT = 2000


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"環境変数 {name} が設定されていません (.env を確認してください)")
    return value


DISCORD_BOT_TOKEN = env_required("DISCORD_BOT_TOKEN")
SARASHINA_API_KEY = env_required("SARASHINA_API_KEY")
CHANNEL_IDS = {
    int(c) for c in os.getenv("DISCORD_CHANNEL_IDS", "").replace(" ", "").split(",") if c
}
DEFAULT_SYSTEM_PROMPT = """あなたは「Sarashina」(SB Intuitions 開発の日本語 AI) です。この Discord サーバーのメンバーの質問や依頼に答えます。
ユーザーの発言は「名前: 本文」の形で渡されます。

回答のルール:
- 直前までの会話の流れを最優先で踏まえる。「それ」「こういうところ」などの指示語や、前の回答への指摘・追加の依頼は、直前のやり取りから解釈する
- 聞かれたことへの答えを最初に、簡潔かつ具体的に書く。依頼された作業 (書き直し・要約など) はそのまま実行する
- 「ご指摘ありがとうございます」「真摯に受け止め」「寄り添う」のような前置き・お礼・謝罪・共感の決まり文句は書かない
- 普通の「です・ます」調で書く。過剰な敬語や、抽象的・大げさな言い回しは使わない
- 質問と関係のない過去の話題を持ち出さない
- 分からないこと・できないことは正直にそう言う。批判されたら言い訳せず、内容で応える
- あなたはインターネットや最新情報にアクセスできない。天気・ニュース・株価などリアルタイムの情報は推測で答えず、分からないと伝える"""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
# 最後のユーザー発言の直後に付ける短い確認 (小さいモデルはプロンプト末尾の指示ほど守るため)
ANSWER_REMINDER = (
    "\n\n---\n(回答前の確認: 前置き・お礼・謝罪は書かず、1文目から答えの本題を書く。"
    "「これ」「こういうところ」などは直前の会話のどこを指しているかを先に特定してから答える)"
)
# モデルが癖で付けがちな決まり文句の書き出し (回答の先頭にあれば取り除く)
STOCK_OPENERS = re.compile(
    r"^(はい、?)?(ご質問の意図を(理解|汲み取り)[^。]*。|ご質問(いただき)?ありがとうございます。|お答え(いた)?します。"
    r"|かしこまりました。|承知(いた)?しました。|ご指摘(いただき)?ありがとうございます。|ご意見を受け取りました。)\s*"
)
RESTATEMENT = re.compile(r"^[^。\n]{0,80}(ということ|について|をお知りになりたい|を知りたい)ですね。\s*")


def strip_stock_openers(text: str) -> str:
    """「ご質問の意図を理解しました。〜についてですね。」のような書き出しを取り除く."""
    original = text
    for _ in range(3):
        stripped = STOCK_OPENERS.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
        text = RESTATEMENT.sub("", text, count=1)
    return text.strip() or original


# 回答時に会話履歴としてそのまま渡す直近の発言数
RECENT_TURNS = int(os.getenv("CONTEXT_RECENT_MESSAGES", "12"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
MODEL = os.getenv("SARASHINA_MODEL", "sarashina3-mini")
CONTEXT_ENABLED = os.getenv("CONTEXT_ENABLED", "true").lower() not in ("0", "false", "no")
CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "sarashina3-nano")
# 思考は nano (60 RPM) が担当し、mini (6 RPM) は最終回答の1回だけに使う
# 速さより回答の質を優先する方針のため既定は on (nano が多段階で考え、その思考メモを mini に渡す)
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "true").lower() not in ("0", "false", "no")
# mini の回答案を nano がチェックし、重大な問題があれば mini が1回だけ書き直す
ANSWER_CHECK_ENABLED = os.getenv("ANSWER_CHECK_ENABLED", "true").lower() not in ("0", "false", "no")
ANSWER_CHECK_PROMPT = """あなたは回答のチェック係です。ユーザーの質問・直前の会話・AI の回答案が与えられます。
手順:
1. まず回答案を見ずに、質問に対する正しい答えの要点を自分で考える (計算や推論は途中式を書いて実際に解く)
2. 自分の答えと回答案を比べ、次の重大な問題があるか判定する
   - 質問に正面から答えていない、または質問の意図を取り違えている
   - 事実・計算・論理の誤り (自分の答えと結論が食い違う)
   - 直前の会話の取り違え (「それ」「こういうところ」などの対象を間違えている等)
   - 根拠のない最新情報 (天気・ニュース等) を断定している
細かい言い回し・文体・好みの問題は指摘しない。
JSON だけを返す: {"own_answer": "自分で考えた答えの要点 (途中式を含む)", "verdict": "ok" または "revise", "problems": ["具体的な問題と直し方", ...]}"""
REVISE_INSTRUCTION = """(チェック係から次の指摘がありました。指摘が正しい点だけを直して、回答全体を書き直してください。
指摘が誤っている場合は元の回答をそのまま出力してください。前置きや「修正しました」等の説明は書かず、回答本文だけを出力すること)
{problems}"""
THINKING_MODEL = os.getenv("THINKING_MODEL", "sarashina3-nano")
# 思考表示の最短時間 (秒). アニメーションを見せる時間も思考に充てる
THINKING_MIN_SECONDS = float(os.getenv("THINKING_MIN_SECONDS", "4"))
# mini の回答枠待ちがこの秒数以上残っていれば「回答設計」パスも行う
THINKING_PLAN_MIN_WAIT = float(os.getenv("THINKING_PLAN_MIN_WAIT", "3"))
THINKING_MAX_CHARS = int(os.getenv("THINKING_MAX_CHARS", "800"))
# true にすると回答後も「💭 X秒 考えました」の1行を残す (既定では回答完了時に消す)
THINKING_KEEP = os.getenv("THINKING_KEEP", "false").lower() in ("1", "true", "yes")
# Discord のメッセージ編集間隔 (秒). 短すぎると Discord 側のレート制限にかかる
STREAM_EDIT_INTERVAL = float(os.getenv("STREAM_EDIT_INTERVAL", "1.2"))

rpm_overrides = {}
if os.getenv("SARASHINA_RPM"):
    rpm_overrides[MODEL] = int(os.environ["SARASHINA_RPM"])
if os.getenv("CONTEXT_RPM"):
    rpm_overrides[CONTEXT_MODEL] = int(os.environ["CONTEXT_RPM"])
if os.getenv("THINKING_RPM"):
    rpm_overrides[THINKING_MODEL] = int(os.environ["THINKING_RPM"])

sarashina = SarashinaClient(
    api_key=SARASHINA_API_KEY,
    model=MODEL,
    max_completion_tokens=int(os.getenv("SARASHINA_MAX_COMPLETION_TOKENS", "2000")),
    temperature=float(os.getenv("SARASHINA_TEMPERATURE", "0.5")),
    rpm=rpm_overrides,
)
# --- チャンネル横断検索 ---
SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "true").lower() not in ("0", "false", "no")
# 索引付けするチャンネル (空なら、対象チャンネルがあるサーバーの Bot が読める全テキストチャンネル)
SEARCH_CHANNEL_IDS = {
    int(c) for c in os.getenv("SEARCH_CHANNEL_IDS", "").replace(" ", "").split(",") if c
} or None
SEARCH_DB = os.getenv("SEARCH_DB", os.path.join(os.path.dirname(__file__), "..", "data", "search.db"))
SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", "5"))
# 検索の回答案を何通り書いて比べるか (mini の n パラメータ. 1リクエスト = レート制限1回分)
ANSWER_CANDIDATES = int(os.getenv("ANSWER_CANDIDATES", "3"))
# 審査で欠けている観点が見つかったら、追加で探して書き直すか (mini をもう1回使う)
ANSWER_REFINE_ENABLED = os.getenv("ANSWER_REFINE_ENABLED", "true").lower() not in ("0", "false", "no")
# 普通の質問でも、役立つ過去の会話をサーバー全体から探して回答に使う (RAG)
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() not in ("0", "false", "no")

search_index = SearchIndex(SEARCH_DB) if SEARCH_ENABLED else None
searcher = Searcher(
    sarashina, search_index, results=SEARCH_RESULTS,
    default_days=float(os.getenv("SEARCH_DEFAULT_DAYS", "365")),
    max_per_author=int(os.getenv("SEARCH_MAX_PER_AUTHOR", "2")),
    min_rounds=int(os.getenv("SEARCH_MIN_ROUNDS", "2")),
    max_rounds=int(os.getenv("SEARCH_MAX_ROUNDS", "6")),
    exhaustive_rounds=int(os.getenv("SEARCH_EXHAUSTIVE_ROUNDS", "5")),
) if search_index else None
indexer: Indexer | None = None

contexts = ContextManager(
    sarashina,
    model=CONTEXT_MODEL,
    fetch_limit=int(os.getenv("CONTEXT_FETCH_LIMIT", "50")),
    max_chars=int(os.getenv("CONTEXT_SUMMARY_CHARS", "600")),
)

intents = discord.Intents.default()
intents.message_content = True


class SarashinaBot(discord.Client):
    async def close(self) -> None:
        await sarashina.close()
        await super().close()


client = SarashinaBot(intents=intents)


def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Discord の2000文字制限に合わせて分割する (なるべく改行で区切る)."""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks or ["(空の応答でした)"]


def clean_content(message: discord.Message) -> str:
    text = message.content
    if client.user:
        text = text.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "")
    return text.strip()


async def build_history(message: discord.Message) -> list[dict]:
    """返信チェーンを遡って会話履歴を組み立てる."""
    chain: list[discord.Message] = []
    current = message
    while current.reference and current.reference.message_id and len(chain) < HISTORY_LIMIT:
        ref = current.reference.resolved
        if not isinstance(ref, discord.Message):
            try:
                ref = await current.channel.fetch_message(current.reference.message_id)
            except discord.HTTPException:
                break
        chain.append(ref)
        current = ref

    messages = []
    for m in reversed(chain):
        role = "assistant" if client.user and m.author.id == client.user.id else "user"
        content = clean_content(m)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def is_target(message: discord.Message) -> bool:
    channel = message.channel
    if channel.id in CHANNEL_IDS:
        return True
    # 対象チャンネル内に作られたスレッドも対象にする
    return isinstance(channel, discord.Thread) and channel.parent_id in CHANNEL_IDS


@client.event
async def on_ready() -> None:
    contexts.bot_user_id = client.user.id if client.user else None
    log.info("ログインしました: %s (対象チャンネル: %s)", client.user, CHANNEL_IDS or "なし")
    if not CHANNEL_IDS:
        log.warning("DISCORD_CHANNEL_IDS が空のため、どのチャンネルにも反応しません")
    if not THINKING_KEEP:
        await cleanup_leftover_thinking()
    global indexer
    if search_index and indexer is None:
        guild_ids = {ch.guild.id for cid in CHANNEL_IDS if (ch := client.get_channel(cid)) and getattr(ch, "guild", None)}
        indexer = Indexer(
            client, sarashina, search_index, SEARCH_CHANNEL_IDS, guild_ids,
            backfill_limit=int(os.getenv("SEARCH_BACKFILL_LIMIT", "5000")),
            backfill_days=float(os.getenv("SEARCH_BACKFILL_DAYS", "365")),
            # Bot と会話するチャンネル自体は取り込まない (SEARCH_INDEX_BOT_CHANNELS=true で取り込む)
            exclude_ids=set() if os.getenv("SEARCH_INDEX_BOT_CHANNELS", "false").lower() in ("1", "true", "yes")
            else set(CHANNEL_IDS),
            interval=float(os.getenv("SEARCH_INDEX_INTERVAL", "120")),
        )
        indexer.start()
        msgs, chunks = search_index.stats()
        log.info("検索インデックスを開始しました (発言 %d 件 / ベクトル化済み %d チャンク / 未処理 %d チャンク)",
                 msgs, chunks, search_index.pending_count())


@client.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
    if search_index:
        search_index.delete_message(payload.message_id)


@client.event
async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent) -> None:
    if search_index and "content" in payload.data:
        search_index.update_message(payload.message_id, payload.data["content"])


async def cleanup_leftover_thinking(limit: int = 100) -> None:
    """再起動などで中断され、チャンネルに残ってしまった思考中の表示を片付ける."""
    if not client.user:
        return
    for channel_id in CHANNEL_IDS:
        try:
            channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
            async for m in channel.history(limit=limit):
                if m.author.id != client.user.id:
                    continue
                content = m.content
                if content[:1] in SPINNER:
                    # 思考中ステータスのまま止まったもの
                    await m.delete()
                elif content.startswith("-# 💭") and not THINKING_KEEP:
                    # 回答途中で止まったもの: 1行目を外し、末尾のカーソルを消す
                    rest = content.split("\n", 1)[1] if "\n" in content else ""
                    rest = rest.removesuffix("▌").rstrip()
                    if rest:
                        await m.edit(content=rest, embeds=[])
                    else:
                        await m.delete()
                elif m.embeds and any((e.title or "").startswith(("💭", "⚠️ 思考")) or (e.title or "")[:1] in SPINNER for e in m.embeds):
                    # 旧バージョンの思考 Embed
                    if content.strip():
                        await m.edit(embeds=[])
                    else:
                        await m.delete()
                else:
                    continue
                log.info("残っていた思考表示を片付けました (message %s)", m.id)
        except discord.HTTPException as e:
            log.warning("思考表示の片付けに失敗しました (channel %s): %s", channel_id, e)


async def thinking_pass(
    view: ThinkingView, title: str, messages: list[dict], max_tokens: int = 1000
) -> str:
    """nano で思考を1パス生成する (内容は Discord には出さず、回答モデルへのメモに積む)."""
    body = await sarashina.chat(
        messages, model=THINKING_MODEL, max_completion_tokens=max_tokens, temperature=0.3
    )
    view.add_section(title, body)
    log.debug("%s:\n%s", title, body)
    return body


async def think(
    view: ThinkingView, context_block: str, history: list[dict], question: str
) -> float:
    """多段階で思考し、mini の回答枠 (予約時刻) を返す.

    mini は 6 RPM しかないため、回答枠が空くまでの待ち時間を nano の追加思考
    (見直し → 回答設計) に充てる。アニメーションを見せる最短時間も思考時間として使う。
    """
    system = THINKING_SYSTEM_PROMPT.format(max_chars=THINKING_MAX_CHARS)
    if context_block:
        system += "\n\n" + context_block
    base = [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]

    def followup(prompt: str) -> list[dict]:
        return [
            *base,
            {"role": "assistant", "content": view.thinking_text},
            {"role": "user", "content": prompt.format(max_chars=THINKING_MAX_CHARS // 2)},
        ]

    # 1. 初回の思考
    view.set_phase("think")
    first = await thinking_pass(view, "思考", base)

    # 2. 見直し (挨拶など思考が短いものは省略)
    if len(first) >= 60:
        view.set_phase("review")
        await thinking_pass(view, "見直し", followup(REVIEW_PROMPT), max_tokens=600)

    # ここで mini の回答枠を予約。空きまでの時間と最短表示時間のうち遅い方まで考え続ける
    slot = sarashina.limiter().reserve()
    target = max(slot, view.started + THINKING_MIN_SECONDS)
    view.deadline = target

    # 3. 回答設計 (待ち時間に余裕があるときだけ)
    if target - time.monotonic() >= THINKING_PLAN_MIN_WAIT:
        view.set_phase("plan")
        await thinking_pass(view, "回答設計", followup(PLAN_PROMPT), max_tokens=600)

    # 4. 残り時間は待機 (アニメーションは動き続ける)
    remaining = target - time.monotonic()
    if remaining > 0.3:
        view.set_phase("wait", cycle=view.thinking_labels())
        await asyncio.sleep(remaining)
    return slot


@client.event
async def on_message(message: discord.Message) -> None:
    if indexer:
        indexer.on_message(message)  # 検索インデックスへのリアルタイム取り込み
    if message.author.bot or not is_target(message):
        return
    question = clean_content(message)
    if not question:
        return

    view = ThinkingView(asker=message.author.display_name)
    view.set_phase("context", f"{message.author.display_name}さんの質問を読んでいます")
    status_msg = await message.reply(view.render(), mention_author=False)
    status = ThrottledEditor(status_msg, STREAM_EDIT_INTERVAL)
    animator = Animator(view, status)
    animator.start()

    # 0. 過去の会話の検索かどうかを振り分け
    request = None
    if searcher:
        try:
            names = [c.name for c in message.guild.text_channels] if message.guild else []
            recent = await contexts.recent_turns(message, 6)
            request = await searcher.route(
                question, names, asker=message.author.display_name, recent=recent,
                guild_id=message.guild.id if message.guild else None,
            )
        except Exception:
            log.exception("振り分けに失敗しました (通常の質問として扱います)")
    if request:
        try:
            answer = await handle_search(message, request, question, view)
        except Exception:
            log.exception("検索に失敗しました")
            answer = "⚠️ 検索中にエラーが発生しました。"
        finally:
            await animator.stop(final=False)
        await post_answer(message, status_msg, answer)
        return

    # 質問に合わせた作業計画 (進捗表示のラベル) を裏で立てる
    plan_task = asyncio.create_task(make_plan(view, question))

    try:
        # 1. チャンネルの背景コンテキスト (要約)
        context_block = ""
        try:
            if CONTEXT_ENABLED:
                ctx = await contexts.get(message)
                context_block = contexts.build_context_block(ctx)
            history = await build_history(message)
            recent = await contexts.recent_turns(message, RECENT_TURNS)
            # 返信先が直近の範囲より古い場合だけ、返信チェーンを前に足す
            recent_text = "\n".join(t["content"] for t in recent)
            history = [h for h in history if h["content"] not in recent_text] + recent
        except Exception:
            log.exception("文脈の取得に失敗しました")
            history = []

        # 1.5 RAG: 質問に役立つ過去の会話をサーバー全体から探す
        knowledge = []
        if searcher and RAG_ENABLED:
            try:
                await asyncio.wait({plan_task}, timeout=3)  # 計画のラベルが間に合えば使う
                view.set_phase("search")
                allowed = searcher.allowed_channels(message.author, message.guild)
                if allowed:
                    knowledge = await searcher.related(
                        question, allowed,
                        exclude_after=time.time() - 3600, exclude_channel=message.channel.id,
                    )
                    if knowledge:
                        log.info("RAG: %d 件の過去の会話を参考にします", len(knowledge))
                        channels = "・".join(dict.fromkeys(f"#{h.messages[0].channel_name}" for h in knowledge))
                        view.note(f"{channels} の過去の会話を{len(knowledge)}件見つけました")
            except Exception:
                log.exception("関連する過去の会話の検索に失敗しました")

        # 2. 思考 (nano) — mini の回答枠が空くまでの時間も思考に使う
        slot = None
        if THINKING_ENABLED:
            try:
                slot = await think(view, context_block, history, question)
            except Exception:
                log.exception("思考の生成に失敗しました (思考なしで回答します)")
        if slot is None:
            slot = await wait_for_mini(view)
    except BaseException:
        await animator.stop(final=False)
        raise

    # 3. 回答 (mini). 生成中もステータスは動かし続け、完成したら新しいメッセージで一気に出す
    view.finish_thinking()
    system_prompt = SYSTEM_PROMPT
    if context_block:
        system_prompt += "\n\n" + context_block
    if knowledge:
        system_prompt += "\n\n" + searcher.knowledge_block(knowledge)
    if view.thinking_text:
        system_prompt += (
            "\n\n## 回答前の思考メモ\n"
            "以下は軽量モデルが事前に考えた下書き(思考・見直し・回答設計)です。方針の参考にしつつ、"
            "計算・事実・論理は自分で検証し、誤りがあれば正してから答えること。"
            "メモ自体や「質問の意図を理解しました」等の前置きには触れず、本題から答える。\n"
            + view.thinking_text
        )
    asker = f"{message.author.display_name}: {question}{ANSWER_REMINDER}"
    if history and history[-1]["role"] == "user":
        history = [*history[:-1], {"role": "user", "content": history[-1]["content"] + "\n" + asker}]
        messages = [{"role": "system", "content": system_prompt}, *history]
    else:
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": asker}]

    try:
        answer = await sarashina.chat(messages, slot=slot)
        if ANSWER_CHECK_ENABLED:
            answer = await check_and_revise(view, messages, answer)
    except SarashinaError as e:
        log.error("Sarashina API エラー: %s", e)
        answer = f"⚠️ Sarashina API でエラーが発生しました: `{truncate(str(e), 300)}`"
    except Exception:
        log.exception("予期しないエラー")
        answer = "⚠️ 予期しないエラーが発生しました。"
    else:
        answer = strip_stock_openers(answer)
        if knowledge:
            answer = searcher.link_knowledge(answer, knowledge)
        if THINKING_KEEP and THINKING_ENABLED:
            answer = f"{view.header}\n{answer}"
    finally:
        plan_task.cancel()
        await animator.stop(final=False)

    await post_answer(message, status_msg, answer)


async def make_plan(view: ThinkingView, question: str) -> None:
    """nano に、この質問に合わせた作業ラベル (集める/考える/確かめる/書く) を作らせる."""
    try:
        raw = await sarashina.chat(
            [{"role": "system", "content": PLAN_LABELS_PROMPT}, {"role": "user", "content": question}],
            model=THINKING_MODEL, max_completion_tokens=300, temperature=0.3,
            response_format={"type": "json_object"},
        )
        view.set_plan(json.loads(raw))
    except Exception:
        log.debug("作業計画の生成に失敗しました", exc_info=True)


async def wait_for_mini(view: ThinkingView, extra: list[str] | None = None, slot: float | None = None) -> float:
    """mini の送信枠を予約し (予約済みならその時刻まで)、空くまで待つ.

    待っている間も「考えている」ことが伝わるよう、質問に合わせた思考ラベルを切り替えて表示する
    (残り時間は進捗バーの横に小さく出る)。
    """
    if slot is None:
        slot = sarashina.limiter().reserve()
    if slot - time.monotonic() > 0.3:
        view.deadline = slot
        view.set_phase("wait", cycle=view.thinking_labels(extra))
        await asyncio.sleep(slot - time.monotonic())
        view.deadline = None
    return slot


async def check_and_revise(view: ThinkingView, messages: list[dict], draft: str) -> str:
    """回答案を nano でチェックし、重大な問題があれば mini で1回だけ書き直す."""
    view.set_phase("check")
    context = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in messages[1:][-5:])
    try:
        raw = await sarashina.chat(
            [
                {"role": "system", "content": ANSWER_CHECK_PROMPT},
                {"role": "user", "content": f"# 質問と直前の会話\n{context}\n\n# 回答案\n{draft}"},
            ],
            model=THINKING_MODEL, max_completion_tokens=800, temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(raw)
    except Exception:
        log.warning("回答のチェックに失敗しました (回答案をそのまま使います)", exc_info=True)
        return draft
    problems = [p for p in result.get("problems") or [] if isinstance(p, str) and p.strip()]
    if result.get("verdict") != "revise" or not problems:
        return draft
    log.info("回答を書き直します: %s", problems)
    view.note("見直しで直したい点が見つかったので書き直しています")
    slot = await wait_for_mini(view, ["指摘された点を確かめています", "直し方を考えています"])
    view.set_phase("revise")
    revised = await sarashina.chat(
        [
            *messages,
            {"role": "assistant", "content": draft},
            {"role": "user", "content": REVISE_INSTRUCTION.format(problems="\n".join(f"- {p}" for p in problems))},
        ],
        slot=slot,
    )
    return revised.strip() or draft


async def post_answer(message: discord.Message, status_msg: discord.Message, answer: str) -> None:
    """回答を新しいメッセージとして投稿してから、ステータスのメッセージを消す."""
    reply = message
    for chunk in split_message(answer):
        reply = await reply.reply(chunk, mention_author=False, suppress_embeds=True)
    try:
        await status_msg.delete()
    except discord.HTTPException as e:
        log.warning("ステータスメッセージの削除に失敗しました: %s", e)


async def handle_search(
    message: discord.Message, req: SearchRequest, question: str, view: ThinkingView
) -> str:
    """過去の会話を検索し、リンク付きの結果 (locate) かまとめ (summarize) を返す."""
    view.topic = searcher._label(req)[:20]
    view.comment(f"「{searcher._label(req)[:16]}」の話を探しています")
    allowed = searcher.allowed_channels(message.author, message.guild)
    if not allowed:
        return "⚠️ 検索できるチャンネルがありません。"
    limit = SEARCH_RESULTS if req.type == "locate" else 6
    # mini の回答枠を先に予約し、空くまでの時間も検索を続けて情報を厚くする
    slot = sarashina.limiter().reserve()

    async def progress(label: str, note: str | None = None) -> None:
        view.comment(label)  # 一言コメント (10秒に1回まで切り替え)

    hits = await searcher.search(
        req, allowed, message.guild.id, limit, question=question, progress=progress, deadline=slot,
    )
    log.info("検索: %s → %d 件 (%s)", req, len(hits), req.type)
    result = ComposeResult("")
    if hits and not all(h.approx for h in hits):
        result = await compose_answer(view, req, hits, question, slot, progress)
        slot = None
    if not result.answer:
        # 通常の検索で答えられなかった → 十分かどうかに関わらず5往復、条件も広げ切って粘り強く探す
        log.info("根拠が足りないため、粘り強く探し直します")
        view.comment("もう少し粘って、言い方や範囲を変えながら探し直しています")
        hits = await searcher.search(
            req, allowed, message.guild.id, limit, question=question, progress=progress, exhaustive=True,
        )
        log.info("粘り強い検索: %d 件 (近い話題のみ=%s)", len(hits), all(h.approx for h in hits))
        if hits and not all(h.approx for h in hits):
            result = await compose_answer(view, req, hits, question, slot, progress)
            slot = None
    if result.answer and result.search and ANSWER_REFINE_ENABLED:
        # 審査で「大事な観点が欠けている」と判定 → その観点で追加検索し、集め直して書き直す (1回まで)
        log.info("欠けている観点「%s」を追加で探します (検索語: %s)", result.missing, result.search)
        view.comment(f"「{result.missing[:20]}」が足りないので、追加で探しています" if result.missing
                     else "足りない観点を追加で探しています")
        extra_req = SearchRequest(req.type, result.search, result.search.split()[:6], person=req.person,
                                  channel=req.channel)
        extra = await searcher.search(extra_req, allowed, message.guild.id, 3, question=question, progress=progress)
        known = {h.chunk_id for h in hits}
        added = [h for h in extra if not h.approx and h.chunk_id not in known]
        if added:
            view.note(f"足りなかった観点について{len(added)}件の会話を追加しました")
            refined = await compose_answer(view, req, hits + added, question, None, progress, refine=True)
            if refined.answer:
                result = refined
    if result.answer:
        return result.answer
    # 5往復探し切っても根拠が無い → 何を補足してほしいか聞き直す
    slot = await wait_for_mini(view, ["探し切っても手がかりが少なかったので、何が分かれば探せるか考えています"], slot=slot)
    view.set_phase("summarize", "聞き直す内容を考えています")
    return await searcher.clarify(req, hits, question, slot=slot)


async def compose_answer(
    view: ThinkingView, req: SearchRequest, hits: list, question: str, slot: float | None,
    progress=None, refine: bool = False,
) -> ComposeResult:
    """見つかった会話から mini が回答案を複数書き、審査で一番良いものを選ぶ (根拠が無ければ空)."""
    if not refine:
        channels = "・".join(dict.fromkeys(f"#{h.messages[0].channel_name}" for h in hits))
        view.note(f"{channels} などで{len(hits)}件の会話が見つかりました")
    slot = await wait_for_mini(view, [f"見つかった{len(hits)}件の会話を読み比べています",
                                      "話の流れを時系列に並べています"], slot=slot)
    view.comment("集めた会話を踏まえて、回答を書き直しています" if refine
                 else f"見つかった{len(hits)}件の会話から回答案を書いています")
    result = await searcher.compose(req, hits, question, slot=slot, progress=progress,
                                    candidates=ANSWER_CANDIDATES)
    result.answer = strip_stock_openers(result.answer) if result.answer else ""
    return result


def setup_logging() -> None:
    """logs/bot.log に日次ローテーションで出力し、LOG_RETENTION_DAYS 日より古いログは自動削除する.

    systemd 経由 (TTY なし) ではコンソールに出さないので journal にログが溜まらない。
    """
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "bot.log"),
        when="midnight",
        backupCount=int(os.getenv("LOG_RETENTION_DAYS", "7")),
        encoding="utf-8",
    )
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    discord.utils.setup_logging(handler=handler, level=level)
    if sys.stderr.isatty():
        discord.utils.setup_logging(level=level)


if __name__ == "__main__":
    setup_logging()
    client.run(DISCORD_BOT_TOKEN, log_handler=None)
