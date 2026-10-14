# CLAUDE.md

## プロジェクト概要
Discord の指定チャンネルの投稿に、国産LLM Sarashina (SB Intuitions) の Chat Completions API で回答する Bot。Python + discord.py。

## 構成
- `bot/main.py` — discord.py Client。`on_message` で対象チャンネル/スレッドを判定し、文脈・履歴を組み立てて回答。起動時 `cleanup_leftover_thinking` で残骸を片付ける
- `bot/sarashina.py` — aiohttp ベースの API クライアント `SarashinaClient`（`chat` / `chat_stream`, `model=` で呼び分け）と、予約制の `RateLimiter`（モデルごとに独立。`reserve()` が送信可能時刻を返し、`chat(slot=...)` でその時刻に送る）
- `bot/context.py` — `ContextManager`。チャンネル/スレッドごとに「要約＋最終取り込みメッセージID＋直近の生発言」をメモリ保持し、質問時に新着分だけを nano で差分要約 → `build_context_block` でシステムプロンプトに付加。Bot のステータス行（スピナーで始まる）は除外
- `bot/thinking.py` — 思考プロンプト（初回/見直し/回答設計）、`ThinkingView`（思考メモの保持と、段階 `PHASES`・スピナー・進捗バーのステータス描画）、`Animator`（バックグラウンドタスクで `STREAM_EDIT_INTERVAL` ごとにステータスを再描画）、`ThrottledEditor`
- 設定はすべてルートの `.env`（`.env.example` 参照）

## on_message の流れ
ステータス行で返信し Animator 開始 → 文脈要約(nano) →（`THINKING_ENABLED` 時のみ `think()`: 思考→見直し→mini 枠 `reserve()`→回答設計/待機）→ mini 枠を予約し埋まっていれば「⏳ 回答の順番を待っています」表示 → `sarashina.chat(slot=...)` で mini の回答を一括生成（この間ステータスは「✍️ 回答を書いています」）→ Animator 停止 → 回答を**新規メッセージ**で投稿 → ステータスのメッセージを削除

## ユーザー要望・決定事項
- 回答はストリーミングせず一気に出す。ステータス（思考ブロック）に上書きせず新規メッセージで投稿する
- Discord には思考の中身を出さず、ステータス（スピナー＋見出し＋進捗バー）だけ表示。Embed は使わない
- 回答後チャンネルには回答だけを残す（ログを重くしない）
- mini の 6 RPM を思考で消費しない（mini 自身に思考させる inline 方式は廃止済み）
- 思考フェーズは既定 off（A/B 検証で mini 単体 9/10 vs 思考あり 10/10 と差が小さく、ユーザー判断で一旦なし）。`THINKING_ENABLED=true` で復活
- サーバーのログは `logs/bot.log`（TimedRotatingFileHandler, 日次, `LOG_RETENTION_DAYS` 日保持）。TTY が無い(systemd)時はコンソール出力しないので journal に溜まらない

## Discord の罠
- 連続編集すると最終編集（特に `embed=None` による削除）が後から元に戻ることがある → 回答は編集ではなく新規投稿＋ステータス削除で出す
- ステータスの編集は Animator だけが行う。削除前に必ず `animator.stop()` する
- メッセージ編集は 1.2 秒間隔程度に抑える（Discord のレート制限）

## 開発環境
- システムに pip/venv が無いため `uv`（~/.local/bin/uv）を使用: `uv venv .venv && uv pip install -p .venv -r requirements.txt`
- 本番稼働: systemd ユーザーサービス `sarashina-bot`（`sarashina-bot.service` を ~/.config/systemd/user にシンボリックリンク、linger 有効）
  - 操作: `systemctl --user {status,restart,stop,disable --now} sarashina-bot` / ログ: `tail -f logs/bot.log`
  - Claude Code のシェルから操作する時は `export XDG_RUNTIME_DIR=/run/user/$(id -u)` が必要
- 手動起動: `./run.sh`（サービス稼働中に実行すると二重応答になるので注意）

## Sarashina API の注意点
- エンドポイント: `POST https://api.sarashina.ai/v1/chat/completions`（Bearer 認証）
- OpenAI とほぼ同形式だが、**ドキュメントに無いフィールドは 400**（`max_tokens` は不可 → `max_completion_tokens` を使う。`logprobs`, `stream_options` も不可）
- `max_completion_tokens` の上限は 8000
- 無償ベータのレート制限: mini/guard/translate = 6 RPM、nano/embedding/rerank = 60 RPM。429 に `Retry-After` は無い
- 長考（reasoning）モデルは無い。mini の方が nano より明確に賢い（AIME 2025: 0.60 vs 0.38）
- `GET /v1/models` は 404
- `<think>` タグは API 側で除去される（思考を区切りたい時は別の見出しを使う）
- `stream: true` は OpenAI 互換の SSE（`data: {...choices[0].delta.content}` / `data: [DONE]`）。`chat_stream` で対応済みだが現在 Bot では未使用
