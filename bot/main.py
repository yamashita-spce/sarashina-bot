"""対象チャンネルの投稿に Sarashina が回答する Discord Bot."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler

import discord
from dotenv import load_dotenv

from context import ContextManager
from sarashina import SarashinaClient, SarashinaError
from thinking import (
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
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "あなたはDiscordサーバーで皆の質問に答える親切なアシスタントです。簡潔かつ正確に日本語で回答してください。",
)
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
MODEL = os.getenv("SARASHINA_MODEL", "sarashina3-mini")
CONTEXT_ENABLED = os.getenv("CONTEXT_ENABLED", "true").lower() not in ("0", "false", "no")
CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "sarashina3-nano")
# 思考は nano (60 RPM) が担当し、mini (6 RPM) は最終回答の1回だけに使う
# 既定は off (A/B 検証で mini 単体との差が小さく、遅延と nano 呼び出しのコストに見合わなかったため)
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() in ("1", "true", "yes")
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
    temperature=float(os.getenv("SARASHINA_TEMPERATURE", "0.7")),
    rpm=rpm_overrides,
)
contexts = ContextManager(
    sarashina,
    model=CONTEXT_MODEL,
    fetch_limit=int(os.getenv("CONTEXT_FETCH_LIMIT", "50")),
    recent_raw=int(os.getenv("CONTEXT_RECENT_MESSAGES", "5")),
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
        view.set_phase("wait")
        await asyncio.sleep(remaining)
    return slot


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or not is_target(message):
        return
    question = clean_content(message)
    if not question:
        return

    view = ThinkingView()
    status_msg = await message.reply(view.render(), mention_author=False)
    status = ThrottledEditor(status_msg, STREAM_EDIT_INTERVAL)
    animator = Animator(view, status)
    animator.start()

    try:
        # 1. チャンネルの背景コンテキスト (要約)
        context_block = ""
        try:
            if CONTEXT_ENABLED:
                ctx = await contexts.get(message)
                context_block = contexts.build_context_block(ctx)
            history = await build_history(message)
        except Exception:
            log.exception("文脈の取得に失敗しました")
            history = []

        # 2. 思考 (nano) — mini の回答枠が空くまでの時間も思考に使う
        slot = None
        if THINKING_ENABLED:
            try:
                slot = await think(view, context_block, history, question)
            except Exception:
                log.exception("思考の生成に失敗しました (思考なしで回答します)")
        if slot is None:
            slot = sarashina.limiter().reserve()
            if slot - time.monotonic() > 0.3:
                view.deadline = slot
                view.set_phase("wait" if THINKING_ENABLED else "queue")
                await asyncio.sleep(slot - time.monotonic())
    except BaseException:
        await animator.stop(final=False)
        raise

    # 3. 回答 (mini). 生成中もステータスは動かし続け、完成したら新しいメッセージで一気に出す
    view.finish_thinking()
    system_prompt = SYSTEM_PROMPT
    if context_block:
        system_prompt += "\n\n" + context_block
    if view.thinking_text:
        system_prompt += (
            "\n\n## 回答前の思考メモ\n"
            "以下は軽量モデルが事前に考えた下書き(思考・見直し・回答設計)です。方針の参考にしつつ、"
            "計算・事実・論理は自分で検証し、誤りがあれば正してから答えること。"
            "メモ自体や「質問の意図を理解しました」等の前置きには触れず、本題から答える。\n"
            + view.thinking_text
        )
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": question}]

    try:
        answer = await sarashina.chat(messages, slot=slot)
    except SarashinaError as e:
        log.error("Sarashina API エラー: %s", e)
        answer = f"⚠️ Sarashina API でエラーが発生しました: `{truncate(str(e), 300)}`"
    except Exception:
        log.exception("予期しないエラー")
        answer = "⚠️ 予期しないエラーが発生しました。"
    else:
        if THINKING_KEEP and THINKING_ENABLED:
            answer = f"{view.header}\n{answer}"
    finally:
        await animator.stop(final=False)

    # 回答を新しいメッセージとして投稿してから、ステータスのメッセージを消す
    reply = message
    for chunk in split_message(answer):
        reply = await reply.reply(chunk, mention_author=False)
    try:
        await status_msg.delete()
    except discord.HTTPException as e:
        log.warning("ステータスメッセージの削除に失敗しました: %s", e)


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
