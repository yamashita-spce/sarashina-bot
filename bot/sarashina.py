"""Sarashina API の非同期クライアント (Chat Completions / Embeddings / Rerank)."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator

import aiohttp

BASE_URL = "https://api.sarashina.ai/v1"
API_URL = f"{BASE_URL}/chat/completions"
EMBED_MODEL = "sarashina3-embedding"
RERANK_MODEL = "sarashina3-rerank"

# 無償ベータのモデル別レート制限 (requests/min)
DEFAULT_RPM = {
    "sarashina3-mini": 6,
    "sarashina3-nano": 60,
    "sarashina3-guard": 6,
    EMBED_MODEL: 60,
    RERANK_MODEL: 60,
}


class SarashinaError(Exception):
    pass


class RateLimiter:
    """1分あたりのリクエスト数を守るため、送信枠を「予約」で割り当てる.

    予約制なので、質問ごとに「自分の番がいつ来るか」を事前に知ることができ、
    待ち時間を思考時間に回せる。
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, rpm)
        self._slots: deque[float] = deque()  # 予約済み送信時刻 (monotonic, 昇順)

    def _prune(self, now: float) -> None:
        while self._slots and self._slots[0] <= now - 60:
            self._slots.popleft()

    def _next_slot(self, now: float) -> float:
        self._prune(now)
        if len(self._slots) < self.rpm:
            t = now
        else:
            t = self._slots[-self.rpm] + 60
        return max(t, self._slots[-1]) if self._slots else t

    def load(self) -> int:
        """直近1分間 (予約済みを含む) の送信数."""
        self._prune(time.monotonic())
        return len(self._slots)

    async def wait_for_headroom(self, headroom: int) -> None:
        """空き枠が headroom 個より多くなるまで待つ (バックグラウンド処理用の低優先度待機)."""
        while self.load() >= self.rpm - headroom:
            await asyncio.sleep(1.0)

    def queue_wait(self) -> float:
        """今予約した場合の待ち秒数 (予約はしない)."""
        now = time.monotonic()
        return max(0.0, self._next_slot(now) - now)

    def reserve(self) -> float:
        """送信枠を予約し、送信してよい時刻 (monotonic) を返す."""
        t = self._next_slot(time.monotonic())
        self._slots.append(t)
        return t

    async def acquire(self) -> None:
        await wait_until(self.reserve())


async def wait_until(t: float) -> None:
    delay = t - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


class SarashinaClient:
    def __init__(
        self,
        api_key: str,
        model: str = "sarashina3-mini",
        max_completion_tokens: int = 2000,
        temperature: float = 0.7,
        rpm: dict[str, int] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._rpm = {**DEFAULT_RPM, **(rpm or {})}
        self._limiters: dict[str, RateLimiter] = {}
        self._session: aiohttp.ClientSession | None = None

    def limiter(self, model: str | None = None) -> RateLimiter:
        """モデルごとに独立したレートリミッターを返す."""
        model = model or self.model
        if model not in self._limiters:
            self._limiters[model] = RateLimiter(self._rpm.get(model, 6))
        return self._limiters[model]

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=180),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _payload(
        self,
        messages: list[dict],
        model: str,
        max_completion_tokens: int | None,
        temperature: float | None,
        stream: bool = False,
        response_format: dict | None = None,
    ) -> dict:
        # 注意: Sarashina API は未定義フィールド (max_tokens, logprobs, stream_options 等) を 400 で拒否する
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens or self.max_completion_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if stream:
            payload["stream"] = True
        if response_format:
            payload["response_format"] = response_format
        return payload

    async def _post(
        self, payload: dict, retries: int, slot: float | None = None, url: str = API_URL
    ) -> aiohttp.ClientResponse:
        """レート制限を守って POST し、200 のレスポンスを返す (429/5xx はリトライ).

        slot: 事前に limiter().reserve() で予約した送信時刻。省略時はここで予約する。
        """
        session = await self._get_session()
        limiter = self.limiter(payload["model"])
        for attempt in range(retries + 1):
            if attempt == 0 and slot is not None:
                await wait_until(slot)
            else:
                await limiter.acquire()
            resp = await session.post(url, json=payload)
            if resp.status == 200:
                return resp
            text = await resp.text()
            resp.release()
            if resp.status == 429 and attempt < retries:
                # Retry-After ヘッダーが無いので次の1分枠まで待つ
                await asyncio.sleep(60)
                continue
            if resp.status >= 500 and attempt < retries:
                await asyncio.sleep(5)
                continue
            raise SarashinaError(f"HTTP {resp.status}: {text[:300]}")
        raise SarashinaError("リトライ上限に達しました")

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
        slot: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        payload = self._payload(
            messages, model or self.model, max_completion_tokens, temperature,
            response_format=response_format,
        )
        resp = await self._post(payload, retries, slot)
        async with resp:
            body = await resp.json()
        content = body["choices"][0]["message"].get("content") or ""
        return content.strip()

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
        slot: float | None = None,
    ) -> AsyncIterator[str]:
        """SSE でストリーミングし、生成されたテキスト断片を順に yield する."""
        payload = self._payload(
            messages, model or self.model, max_completion_tokens, temperature, stream=True
        )
        resp = await self._post(payload, retries, slot)
        async with resp:
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def embed(self, text: str, task_type: str, retries: int = 2) -> list[float]:
        """テキストを埋め込みベクトル (1536次元) に変換する. 1リクエスト1テキストのみ."""
        payload = {"model": EMBED_MODEL, "input": text, "task_type": task_type}
        resp = await self._post(payload, retries, url=f"{BASE_URL}/embeddings")
        async with resp:
            body = await resp.json()
        return body["data"][0]["embedding"]

    async def rerank(
        self, query: str, documents: list[str], task_type: str = "GENERAL_RETRIEVAL", retries: int = 2
    ) -> list[tuple[int, float]]:
        """documents (最大20件) を query との関連度順に並べ替え、(元の index, score) を返す."""
        if not documents:
            return []
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents[:20],
            "task_type": task_type,
        }
        resp = await self._post(payload, retries, url=f"{BASE_URL}/rerank")
        async with resp:
            body = await resp.json()
        return [(r["document_id"], r["score"]) for r in body["data"][0]["result"]]
