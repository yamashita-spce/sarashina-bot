# Sarashina Discord Bot

国産LLM **Sarashina** (SB Intuitions) の API を使い、Discord の指定チャンネルに投稿された質問へ自動で回答する Bot です。

## 機能

- 指定チャンネル（とその中のスレッド）への投稿に、Sarashina（`sarashina3-mini`）が返信で回答
- **チャンネルの文脈を踏まえて回答**：質問のたびにチャンネルの直近の会話を `sarashina3-nano` に要約させ、その要約＋直近の発言数件を背景コンテキストとしてプロンプトに加えてから回答する（「それっていつだっけ？」のような質問にも答えられる）
- **処理中ステータスのアニメーション表示**：スピナー・切り替わる見出し・進捗バーで「文脈を読んでいる」「回答の順番待ち（あと約N秒）」「回答を書いています」などを表示
- **回答は一気に表示**：回答が完成したら新しいメッセージとして一度に投稿し、ステータスのメッセージは削除する（ステータスに上書きしない）。チャンネルには回答だけが残る
- **mini のレートリミット対策**：mini（6 RPM）の送信枠を予約制で管理し、混雑時は何秒後に回答できるかを表示する
- **思考フェーズ（既定 off）**：`THINKING_ENABLED=true` で、回答前に nano がバックエンドで多段階に考え、その思考メモを mini に渡す（Discord には思考の中身は表示しない）
- Bot の回答に「返信」すると、返信チェーンを会話履歴として引き継いで続きの会話ができる
- 2000文字を超える回答は分割して投稿

## セットアップ

### 1. Sarashina API キー

https://chat.sarashina.ai/dashboard/api-keys で発行し、`.env` の `SARASHINA_API_KEY` に設定します。

### 2. Discord Bot の作成

1. [Discord Developer Portal](https://discord.com/developers/applications) で New Application を作成
2. **Bot** タブで Token を発行 → `.env` の `DISCORD_BOT_TOKEN` に設定
3. 同じ Bot タブで **MESSAGE CONTENT INTENT** を ON にする（必須）
4. **OAuth2 > URL Generator** で scope `bot` を選び、権限に
   `View Channels` / `Send Messages` / `Send Messages in Threads` / `Read Message History` / `Add Reactions` を付けて URL を生成し、サーバーに招待
5. Discord の「開発者モード」を ON にし、対象チャンネルを右クリック →「チャンネルIDをコピー」→ `.env` の `DISCORD_CHANNEL_IDS` に設定（複数はカンマ区切り）

### 3. インストールと起動

```bash
cp .env.example .env   # 値を設定
uv venv .venv
uv pip install -p .venv -r requirements.txt
./run.sh
```

## 常時起動 (systemd)

ユーザー単位の systemd サービスとして登録済みです（PC起動時に自動起動、落ちても10秒後に自動再起動）。
`loginctl enable-linger` 設定済みのため、ログアウト中も動き続けます。

```bash
systemctl --user status sarashina-bot     # 状態確認
tail -f logs/bot.log                      # ログをリアルタイム表示 (Ctrl+C で抜ける)
systemctl --user restart sarashina-bot    # 再起動 (.env やコードを変更したとき)
```

### ログ

ログは `logs/bot.log` に出力され、毎日0時にローテーション、`LOG_RETENTION_DAYS`（既定7日）より古いものは自動削除されます。systemd の journal には起動・停止の記録とクラッシュ時のエラーしか残りません。

```bash
tail -f logs/bot.log    # ログをリアルタイム表示
```

### 停止する方法

```bash
systemctl --user stop sarashina-bot       # 一時停止 (PC再起動すると、また自動起動する)
systemctl --user disable --now sarashina-bot  # 停止＋自動起動もやめる
systemctl --user enable --now sarashina-bot   # 再び常時起動に戻す
```

### 新しい環境でサービスを登録する場合

```bash
mkdir -p ~/.config/systemd/user
ln -sf "$PWD/sarashina-bot.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sarashina-bot
loginctl enable-linger "$USER"   # ログアウト中も動かす
```

※ `sarashina-bot.service` 内のパスは `/home/shita/private/salashina` 固定です。別の場所に置く場合は書き換えてください。

## 設定項目 (.env)

| 変数 | 必須 | 既定値 | 説明 |
|---|---|---|---|
| `SARASHINA_API_KEY` | ✅ | | Sarashina API キー |
| `DISCORD_BOT_TOKEN` | ✅ | | Discord Bot トークン |
| `DISCORD_CHANNEL_IDS` | ✅ | | 反応するチャンネルID（カンマ区切り） |
| `SARASHINA_MODEL` | | `sarashina3-mini` | `sarashina3-nano` にすると軽量・高速（60 RPM） |
| `SARASHINA_MAX_COMPLETION_TOKENS` | | `2000` | 最大出力トークン（上限 8000） |
| `SARASHINA_TEMPERATURE` | | `0.7` | 0.0〜2.0 |
| `SARASHINA_RPM` | | `6` | 1分あたりのリクエスト上限 |
| `SYSTEM_PROMPT` | | 日本語アシスタント | システムプロンプト |
| `HISTORY_LIMIT` | | `10` | 返信チェーンを遡る最大件数 |
| `CONTEXT_ENABLED` | | `true` | チャンネル文脈の要約を使うか |
| `CONTEXT_MODEL` | | `sarashina3-nano` | 要約に使うモデル |
| `CONTEXT_RPM` | | モデル既定値 | 要約モデルのリクエスト上限/分 |
| `CONTEXT_FETCH_LIMIT` | | `50` | 起動後初回に読み込む過去メッセージ数 |
| `CONTEXT_RECENT_MESSAGES` | | `5` | 要約と一緒にそのまま渡す直近の発言数 |
| `CONTEXT_SUMMARY_CHARS` | | `600` | 要約の目安文字数 |
| `THINKING_ENABLED` | | `false` | 思考フェーズを行うか（下記「思考フェーズの効果検証」参照） |
| `THINKING_KEEP` | | `false` | 思考フェーズ有効時、`true` で回答の先頭に「💭 X秒 考えました」の1行を付ける |
| `THINKING_MODEL` | | `sarashina3-nano` | 思考モデル |
| `THINKING_RPM` | | モデル既定値 | 思考モデルのリクエスト上限/分 |
| `THINKING_MAX_CHARS` | | `800` | 初回思考の目安文字数（見直し・回答設計はその半分。Discord には表示されない） |
| `THINKING_MIN_SECONDS` | | `4` | 思考表示の最短時間（アニメーションを見せる時間も思考に充てる） |
| `THINKING_PLAN_MIN_WAIT` | | `3` | mini の回答枠待ちがこの秒数以上残っていれば「回答設計」パスも行う |
| `LOG_RETENTION_DAYS` | | `7` | ログの保存日数（これより古いログは自動削除） |
| `LOG_LEVEL` | | `INFO` | ログレベル |
| `STREAM_EDIT_INTERVAL` | | `1.2` | ステータスのアニメーションでメッセージを編集する間隔(秒) |

### チャンネル文脈の仕組み

1. 質問が来ると、前回要約した時点以降の新着発言を取得（起動直後は直近 `CONTEXT_FETCH_LIMIT` 件）
2. 「これまでの要約＋新着発言」を `sarashina3-nano` に渡して要約を更新（チャンネル/スレッドごとにメモリ保持、新着が無ければ再要約しない）
3. 要約と直近の発言をシステムプロンプトに追加し、`sarashina3-mini` で回答

### Discord 上での表示

質問すると Bot はまずステータスのメッセージで返信し、約1.2秒ごとに表示が切り替わります。

```
⠋ 📚 チャンネルの文脈を読み解いています.
-# ▰▱▱▱▱▱▱▱▱▱▱▱▱▱ 0秒
⠙ ⏳ 回答の順番を待っています..              ← mini の枠が埋まっている時だけ
-# ▰▰▰▰▱▱▱▱▱▱▱▱▱▱ 4秒  あと約8秒
⠹ ✍️ 回答を書いています...
-# ▱▱▰▰▰▱▱▱▱▱▱▱▱▱ 13秒
→ 回答が完成したら新しいメッセージで一気に投稿し、このステータスのメッセージは削除
```

- 見出しは段階と時間（数秒ごと）で切り替わり、スピナー・末尾の「.」も動く
- 進捗バーは、終了予定が分かっている間（順番待ち）は残り時間に応じて伸び、それ以外は流れるアニメーション
- 回答はステータスを編集せず新規メッセージで出す（Discord は連続編集すると最終編集が元に戻ることがあるため）
- 再起動などで処理が中断されステータスが残った場合は、Bot 起動時に対象チャンネルの直近100件を確認して自動で片付ける
- ステータスのメッセージはチャンネル文脈の要約には含めない

### mini のレートリミット対策

- 送信枠を予約制で管理する（`RateLimiter.reserve`）。質問ごとに「自分の番が何秒後か」が分かるので、ステータスに残り秒数を表示し、予約時刻ちょうどに送信する
- 予約時刻に送るため、通常は 429 エラーにならない（万一 429 になった場合は1分待ってリトライ）

### 思考フェーズ（`THINKING_ENABLED=true` のとき）

- Sarashina API には長考（reasoning）モデルが無い。mini（AIME 2025: 0.60）は nano（0.38）より賢いが 6 RPM しかないため、思考は 60 RPM の nano に任せ、mini は最終回答の1回だけに使う
- nano が「思考（意図→文脈→分解して検討→検算→方針）→ 見直し →（mini の順番待ちに余裕があれば）回答設計」と多段階で考え、その思考メモを mini のシステムプロンプトに渡す。mini には「下書きなので自分で検証し、誤りは正す」よう指示
- mini の順番待ちの時間を nano の追加思考に充てる。ステータスは 💭 思考 → 🔁 見直し → 🧭 回答設計 → ⏳ 待機 と切り替わる。最短 `THINKING_MIN_SECONDS` 秒は考える

効果検証（2026-10-14）:
- 正解が明確な10問（計算・論理・文字数え等）で「mini 単体」と「nano 思考＋mini」を比較 → mini 単体 9/10、思考あり 10/10（差は strawberry の r の数え上げ1問のみ）
- 効果は小さく有意とは言えない一方、遅延と nano 呼び出しが増えるため、**既定では off**

### API 呼び出し回数

1質問あたり「要約(nano, 新着がある時) + 回答(mini, 1回)」。思考フェーズ有効時は nano が最大3回増える。

## 構成

```
bot/main.py       Discord Bot 本体（メッセージ受信・履歴構築・思考・回答投稿・残骸の片付け）
bot/sarashina.py  Sarashina API クライアントとモデル別レートリミッター
bot/context.py    チャンネル会話の要約（背景コンテキスト）管理
bot/thinking.py   思考プロンプト、処理中ステータスの描画（ThinkingView）、アニメーション（Animator）、編集間隔制御（ThrottledEditor）
logs/             ログ（日次ローテーション、自動削除）
run.sh            手動起動スクリプト（systemd 稼働中は二重起動しないこと）
sarashina-bot.service  systemd ユーザーサービス定義
```

## 参考

- Sarashina API ドキュメント: https://docs.sarashina.ai/
