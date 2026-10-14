# CLAUDE.md

## プロジェクト概要
Discord の指定チャンネルの投稿に、国産LLM Sarashina (SB Intuitions) の Chat Completions API で回答する Bot。Python + discord.py。

## 構成
- `bot/main.py` — discord.py Client。`on_message` で対象チャンネル/スレッドを判定し、文脈・履歴を組み立てて回答。起動時 `cleanup_leftover_thinking` で残骸を片付ける
- `bot/sarashina.py` — aiohttp ベースの API クライアント `SarashinaClient`（`chat` / `chat_stream`, `model=` で呼び分け）と、予約制の `RateLimiter`（モデルごとに独立。`reserve()` が送信可能時刻を返し、`chat(slot=...)` でその時刻に送る）
- `bot/context.py` — `ContextManager`。チャンネル/スレッドごとに「要約＋最終取り込みメッセージID＋直近の生発言」をメモリ保持し、質問時に新着分だけを nano で差分要約 → `build_context_block` でシステムプロンプトに付加。Bot のステータス行（スピナーで始まる）は除外
- `bot/thinking.py` — 思考プロンプト（初回/見直し/回答設計）、`ThinkingView`（思考メモの保持と、段階 `PHASES`・スピナー・進捗バーのステータス描画）、`Animator`（バックグラウンドタスクで `STREAM_EDIT_INTERVAL` ごとにステータスを再描画）、`ThrottledEditor`
- `bot/search.py` — `Searcher`。`route()`(nano で検索判定＋条件抽出, JSON mode) → `search()`(embedding ベクトル検索＋LIKE キーワード検索 → rerank → nano 関連判定で会話と中心発言を選ぶ。期間指定で0件なら期間なしで再検索) → `format_locate()`(リンク付き一覧) / `summarize()`(mini で要約、`[n]` を jump_url リンクに変換)。`allowed_channels()` で質問者が見えるチャンネルに限定
- `bot/search_index.py` — `SearchIndex`(SQLite: messages/chunks/channel_state。チャンクは保存時は embedding=NULL、後で `set_embedding`。ベクトルは 256 次元に切り詰め正規化し numpy 行列でメモリ保持) と `Indexer`(3つのタスク: `_crawl_loop`=取りこぼし補完 `catch_up`＋過去ログ `backfill`（新しいチャンネル順・1年まで）、`_flush_loop`=`on_message` で受けた新着を会話が3分途切れたら保存、`_embed_loop`=未処理チャンクを新しい順にベクトル化（`wait_for_headroom(20)`）)
- 設定はすべてルートの `.env`（`.env.example` 参照）

## on_message の流れ
ステータス行で返信し Animator 開始 → `searcher.route()` で検索判定（検索なら `handle_search` → 結果を新規投稿してステータス削除で終了）→ 文脈要約(nano) →（`THINKING_ENABLED` 時のみ `think()`: 思考→見直し→mini 枠 `reserve()`→回答設計/待機）→ mini 枠を予約し埋まっていれば「⏳ 回答の順番を待っています」表示 → `sarashina.chat(slot=...)` で mini の回答を一括生成（この間ステータスは「✍️ 回答を書いています」）→ Animator 停止 → 回答を**新規メッセージ**で投稿 → ステータスのメッセージを削除

## ユーザー要望・決定事項
- 回答はストリーミングせず一気に出す。ステータス（思考ブロック）に上書きせず新規メッセージで投稿する
- Discord には思考の中身を出さず、ステータス（スピナー＋見出し＋進捗バー）だけ表示。Embed は使わない
- 回答後チャンネルには回答だけを残す（ログを重くしない）
- mini の 6 RPM を思考で消費しない（mini 自身に思考させる inline 方式は廃止済み）
- 思考フェーズは既定 off（A/B 検証で mini 単体 9/10 vs 思考あり 10/10 と差が小さく、ユーザー判断で一旦なし）。`THINKING_ENABLED=true` で復活
- サーバーのログは `logs/bot.log`（TimedRotatingFileHandler, 日次, `LOG_RETENTION_DAYS` 日保持）。TTY が無い(systemd)時はコンソール出力しないので journal に溜まらない

## 回答品質の設計判断（実際の不具合から）
- 直近の会話を要約だけで渡すと流れを取り違える（「こうゆうところ」を誤解、過去の話題=釣りを無関係な回答に持ち出す）→ `ContextManager.recent_turns` で直近12件を生の会話ターンとして渡し、要約は「関係する場合だけ使う背景」に格下げ
- mini はシステムプロンプトの文体指示を守りにくい → 最後のユーザー発言に `ANSWER_REMINDER` を付け、`strip_stock_openers` で決まり文句の書き出しを除去
- 検索の振り分けに会話の流れを渡さないと、追加の依頼（「古すぎる」「直近のを」）を解釈できず、プロンプト例の人名をそのまま使ってしまった → 直前6件を渡し、例は匿名化
- Bot への感想が検索扱いになった → `SEARCH_CUES` を含まない投稿は nano を呼ばず chat。order=recent の持ち越しや勝手な until 設定はコード側で補正
- 新しさの重みを日数の半減期で計算すると、会話がまばらなサーバーでは効かない → 結果内の相対順位で評価（関連度 6 : 新しさ 4）

## RAG（普通の質問）
- `Searcher.related()`: embedding 候補15 → rerank 上位6 → nano (`KNOWLEDGE_PROMPT`) で役立つものだけ（最大3）→ `knowledge_block` で `[参N]` 付きでシステムプロンプトへ → `link_knowledge` で `[参N]` をリンク化し末尾に出典
- Sarashina API にマネージドな検索（File Search / Vector Store 相当）は無い。部品の組み合わせで RAG を作る方針（ユーザー要望）
- 天気など最新情報を RAG の無関係なリンクから捏造した例があった → URL だけの発言は根拠にしない＋「最新情報は推測しない」をシステムプロンプトに追加

## 検索の偏り対策（ユーザー指摘: 古すぎる・特定の人に偏る）
- 取り込みは保存とベクトル化を分離（保存は API 不要で全チャンネル約2分、ベクトル化は新しい順に約40件/分）
- 期間指定なしの検索は直近1年 → 0件なら全期間。1人あたり2件まで（`_diversify`）
- 英字キーワードは単語境界で照合（`keyword_pattern`）。nano が落とした候補はクエリ語句そのものを含む場合だけ復活

## 検索の設計判断
- 埋め込みは発言単位ではなく会話チャンク単位（短い発言は意味が取りにくい・埋め込み 60 RPM で全履歴は重い）
- rerank のスコアは無関係でも 0.9 超になることがあり、しきい値に使えない → nano で関連判定（実データで確認済み）
- 実データは URL・画像・相づちが多く、発言単位の rerank ではリンク先選びが不正確 → nano に会話全体を読ませて中心の発言を選ばせる
- `numpy` を使用（requirements.txt）

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
