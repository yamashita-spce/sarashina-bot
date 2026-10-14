"""回答前の「思考」(バックエンド) と、Discord 上の処理中ステータス表示."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import discord

log = logging.getLogger("sarashina-bot.thinking")

THINKING_SYSTEM_PROMPT = """あなたはアシスタントの「思考」担当です。別の回答担当モデルが最終回答を書く前に、問題を考え抜いて思考メモを作ってください。

手順:
1. 意図の確認: 質問者が本当に知りたいことは何か。曖昧なら考えられる解釈を挙げ、最も妥当なものを選ぶ
2. 文脈の確認: チャンネルの背景情報・会話履歴のうち、この質問に関係する事実や条件を拾う(前の発言の「それ」「さっきの」が何を指すか等)
3. 検討: 問題を小さく分解し、一歩ずつ考える。計算・論理パズル・比較検討は途中式や根拠を書きながら実際に解く
4. 検算: 出した結論を条件に当てはめて確かめる。誤りや見落とし、別解がないか疑う
5. 方針: 最終回答に含めるべき要点と構成、注意点(不確かな点・知識が古い可能性など)

挨拶や雑談など簡単なものは1〜2行でよい。{max_chars}字以内、独り言のような箇条書きで書くこと。
ユーザー向けの最終回答の文章そのものは書かないこと。"""

# 思考の追加パス (待ち時間があるときに順に実行する)
REVIEW_PROMPT = """上のあなたの思考メモを、別人の目で批判的に見直してください。
計算・事実・論理の誤り、見落とした条件、別の解釈や反例がないかを検証し、問題があれば修正した結論を書いてください。
問題がなければ、その根拠を一言添えて「問題なし」と書いてください。{max_chars}字以内の箇条書きで。"""

PLAN_PROMPT = """ここまでの思考と見直しを踏まえ、最終回答の設計図を作ってください。
- 冒頭で伝える結論
- 構成(順番・見出し)と、必ず含める要点
- 質問者のレベルに合わせた説明の深さ、避けるべき表現
を{max_chars}字以内の箇条書きで。回答本文は書かないこと。"""


@dataclass(frozen=True)
class Phase:
    emoji: str
    labels: tuple[str, ...]  # 時間とともに切り替わる見出し


PHASES = {
    "context": Phase("📚", (
        "チャンネルの文脈を読み解いています",
        "会話の流れを把握しています",
    )),
    "think": Phase("💭", (
        "思考を巡らせています",
        "論点を整理しています",
        "問題を分解しています",
        "一歩ずつ検討しています",
        "仮説を立てています",
    )),
    "review": Phase("🔁", (
        "自分の考えを疑っています",
        "検算しています",
        "別の解釈を探しています",
        "反例を探しています",
    )),
    "plan": Phase("🧭", (
        "回答の構成を練っています",
        "要点を絞り込んでいます",
        "伝え方を考えています",
    )),
    "wait": Phase("⏳", (
        "回答枠の空きを待ちながら熟考中",
        "考えを寝かせています",
        "最終チェック中",
    )),
    "search": Phase("🔍", (
        "チャンネルを検索しています",
        "関連する会話を探しています",
        "過去ログをめくっています",
        "該当する発言を特定しています",
    )),
    "summarize": Phase("📝", (
        "会話をまとめています",
        "要点を整理しています",
    )),
    "queue": Phase("⏳", (
        "回答の順番を待っています",
        "混雑中のため少々お待ちください",
    )),
    "answer": Phase("✍️", ("回答を書いています", "回答をまとめています")),
}

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
LABEL_SECONDS = 3.0  # 見出しを切り替える間隔
BAR_CELLS = 14
CURSOR = " ▌"


def truncate(text: str, limit: int, tail: bool = False) -> str:
    """limit 文字に収める. tail=True なら末尾側を残す (ストリーミング中の表示用)."""
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):] if tail else text[: limit - 1] + "…"


class ThinkingView:
    """思考中ステータス (1行) の状態と描画.

    思考の中身はバックエンドで回答モデルに渡すだけで、Discord には表示しない。
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.phase = "context"
        self.phase_started = self.started
        self.sections: list[tuple[str, str]] = []  # (見出し, 本文) — 回答モデルに渡す思考メモ
        self.deadline: float | None = None  # 思考を終える予定時刻 (進捗バー用)
        self.frame = 0
        self.thinking_seconds: float | None = None

    def set_phase(self, key: str) -> None:
        self.phase = key
        self.phase_started = time.monotonic()

    def add_section(self, title: str, body: str) -> None:
        self.sections.append((title, body.strip()))

    def finish_thinking(self) -> None:
        self.thinking_seconds = time.monotonic() - self.started
        self.deadline = None
        self.set_phase("answer")

    @property
    def thinking_text(self) -> str:
        """回答モデルに渡す思考メモ全体."""
        return "\n\n".join(f"### {t}\n{b}" for t, b in self.sections if b)

    @property
    def header(self) -> str:
        """THINKING_KEEP 時に回答の上に小さく出す1行."""
        return f"-# 💭 {self.thinking_seconds or 0:.1f}秒 考えました"

    def _bar(self, now: float) -> str:
        if self.deadline and self.deadline > self.started:
            ratio = min(1.0, (now - self.started) / (self.deadline - self.started))
            filled = round(ratio * BAR_CELLS)
            return "▰" * filled + "▱" * (BAR_CELLS - filled)
        # 終了予定が未定の間は、光が流れるような不定プログレス
        pos = self.frame % (BAR_CELLS + 3)
        return "".join("▰" if pos - 3 < i <= pos else "▱" for i in range(BAR_CELLS))

    def render(self) -> str:
        now = time.monotonic()
        p = PHASES[self.phase]
        label = p.labels[int((now - self.phase_started) / LABEL_SECONDS) % len(p.labels)]
        spinner = SPINNER[self.frame % len(SPINNER)]
        dots = "." * (self.frame % 3 + 1)
        remain = ""
        if self.deadline and self.deadline - now >= 1:
            remain = f"  あと約{self.deadline - now:.0f}秒"
        return f"{spinner} {p.emoji} {label}{dots}\n-# `{self._bar(now)}` {now - self.started:.0f}秒{remain}"


class ThrottledEditor:
    """Discord の編集レート制限に引っかからないよう、一定間隔でのみメッセージを編集する."""

    def __init__(self, message: discord.Message, interval: float) -> None:
        self.message = message
        self.interval = interval
        self._last = 0.0

    async def edit(self, force: bool = False, **kwargs) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        try:
            await self.message.edit(**kwargs)
        except discord.HTTPException as e:
            log.warning("メッセージの編集に失敗しました: %s", e)


class Animator:
    """一定間隔でステータス行を描き直し、スピナーや見出しを動かし続ける."""

    def __init__(self, view: ThinkingView, editor: ThrottledEditor) -> None:
        self.view = view
        self.editor = editor
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            await self.editor.edit(force=True, content=self.view.render())
            self.view.frame += 1
            await asyncio.sleep(self.editor.interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self, final: bool = True) -> None:
        """アニメーションを止め、final なら現在の状態で最後に1回描画する."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if final:
            await self.editor.edit(force=True, content=self.view.render())
