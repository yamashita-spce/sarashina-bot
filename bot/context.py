"""チャンネルの会話を Sarashina に要約させ、背景コンテキストとして保持する."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import discord

from sarashina import SarashinaClient
from thinking import SPINNER

log = logging.getLogger("sarashina-bot.context")

SUMMARY_SYSTEM_PROMPT = """あなたはDiscordチャンネルの会話記録係です。
「これまでの要約」と「新しい発言」をもとに、チャンネルの背景コンテキストを更新した要約を作成してください。

要件:
- 日本語で{max_chars}字以内
- 現在の話題・前提知識・決定事項・未解決の質問・誰が何に関心を持っているかを優先して残す
- 古くて今後の会話に関係なさそうな話題は削ってよい
- 推測は書かず、会話に書かれている事実だけをまとめる
- 要約本文のみを出力する(前置きや見出しは不要)"""


@dataclass
class ChannelContext:
    summary: str = ""
    last_message_id: int | None = None
    recent: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ContextManager:
    def __init__(
        self,
        client: SarashinaClient,
        bot_user_id: int | None = None,
        model: str = "sarashina3-nano",
        fetch_limit: int = 50,
        recent_raw: int = 5,
        max_chars: int = 600,
        message_chars: int = 400,
    ) -> None:
        self.client = client
        self.bot_user_id = bot_user_id
        self.model = model
        self.fetch_limit = fetch_limit
        self.recent_raw = recent_raw
        self.max_chars = max_chars
        self.message_chars = message_chars
        self._channels: dict[int, ChannelContext] = {}

    def format_message(self, m: discord.Message) -> str | None:
        text = m.clean_content.strip()
        if m.author.id == self.bot_user_id and text[:1] in SPINNER:
            return None  # 処理中ステータスは文脈に含めない
        if m.attachments:
            text += " " + " ".join(f"[添付: {a.filename}]" for a in m.attachments)
        text = text.strip()
        if not text:
            return None
        if len(text) > self.message_chars:
            text = text[: self.message_chars] + "…"
        name = "Sarashina(あなた)" if m.author.id == self.bot_user_id else m.author.display_name
        return f"[{m.created_at.astimezone().strftime('%m/%d %H:%M')}] {name}: {text}"

    async def get(self, message: discord.Message) -> ChannelContext:
        """質問メッセージより前の新着発言を取り込んで要約を更新し、コンテキストを返す."""
        ctx = self._channels.setdefault(message.channel.id, ChannelContext())
        async with ctx.lock:
            if ctx.last_message_id:
                # 前回取り込んだ発言以降の新着のみ
                new_messages = [
                    m
                    async for m in message.channel.history(
                        limit=self.fetch_limit,
                        before=message,
                        after=discord.Object(ctx.last_message_id),
                        oldest_first=True,
                    )
                ]
            else:
                # 初回 (起動直後) は直近 fetch_limit 件
                new_messages = [
                    m async for m in message.channel.history(limit=self.fetch_limit, before=message)
                ][::-1]
            lines = [line for m in new_messages if (line := self.format_message(m))]
            if new_messages:
                ctx.last_message_id = new_messages[-1].id
            if lines:
                ctx.recent = (ctx.recent + lines)[-self.recent_raw :]
                try:
                    ctx.summary = await self._summarize(ctx.summary, lines)
                    log.info("チャンネル %s の要約を更新 (新着 %d 件)", message.channel.id, len(lines))
                except Exception:
                    # 要約に失敗しても回答は続ける (直近の生ログは使える)
                    log.exception("要約の更新に失敗しました")
            return ctx

    async def _summarize(self, previous: str, lines: list[str]) -> str:
        user = f"# これまでの要約\n{previous or '(なし)'}\n\n# 新しい発言\n" + "\n".join(lines)
        return await self.client.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT.format(max_chars=self.max_chars)},
                {"role": "user", "content": user},
            ],
            model=self.model,
            max_completion_tokens=1000,
            temperature=0.2,
        )

    def build_context_block(self, ctx: ChannelContext) -> str:
        """システムプロンプトに追加する、少し前までの会話の要約 (直近のやり取りは会話履歴として別に渡す)."""
        if not ctx.summary:
            return ""
        return (
            "## 参考: このチャンネルの少し前までの会話の要約\n"
            "質問がこれに関係する場合だけ使うこと。関係のない過去の話題を持ち出さない。\n"
            + ctx.summary
        )

    async def recent_turns(self, message: discord.Message, limit: int) -> list[dict]:
        """質問の直前 limit 件の発言を、そのままの会話履歴 (user/assistant のターン) にする.

        他の人の発言は「名前: 本文」の user ターン、Bot 自身の回答は assistant ターン。
        直近のやり取りを最優先で踏まえさせるため、要約ではなく生の発言を渡す。
        """
        msgs = [m async for m in message.channel.history(limit=limit + 5, before=message)][::-1]
        turns: list[dict] = []
        for m in msgs[-limit:]:
            text = m.clean_content.strip()
            if m.author.id == self.bot_user_id:
                if not text or text[:1] in SPINNER:
                    continue  # 処理中ステータスは除外
                role, content = "assistant", text
            else:
                line = self.format_message(m)
                if not line:
                    continue
                role, content = "user", line.split("] ", 1)[1]  # 「名前: 本文」
            if len(content) > 1500:
                content = content[:1500] + "…"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n" + content  # 同じ役割が続く場合はまとめる
            else:
                turns.append({"role": role, "content": content})
        # 会話履歴は user から始める
        while turns and turns[0]["role"] == "assistant":
            turns.pop(0)
        return turns
