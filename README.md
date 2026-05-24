# OpenCV 人臉情緒辨識原型

這個原型會做三件事：

1. 開啟電腦攝影機
2. 偵測人臉
3. 使用 FER+ ONNX 模型預測 7 類基本情緒

## 目前輸出情緒類別

- neutral
- happiness
- sadness
- anger
- disgust
- fear
- contempt

## 執行方式

在工作區根目錄執行：

```powershell
& ".venv/Scripts/python.exe" -m pip install -r requirements.txt
```

然後執行：

```powershell
& ".venv/Scripts/python.exe" emotion_camera.py
```

如果要指定其他攝影機：

```powershell
& ".venv/Scripts/python.exe" emotion_camera.py --camera 1
```

如果你不想要鏡像反轉（預設為開啟）：

```powershell
& ".venv/Scripts/python.exe" emotion_camera.py --no-mirror
```

如果你覺得人臉不夠靈敏，可以再降低最小臉尺寸：

```powershell
& ".venv/Scripts/python.exe" emotion_camera.py --min-face 36
```

如果你覺得情緒結果不穩定，建議用這組參數：

```powershell
& ".venv/Scripts/python.exe" emotion_camera.py --min-face 36 --smooth-alpha 0.22 --confidence-threshold 0.5 --face-padding 0.22 --neutral-penalty 0.5 --emotion-boost 1.3
```

## 15 秒情緒回傳

- 程式會每秒輸出 15 秒統計結果到 emotion_output/latest_emotion.json
- 欄位包含 dominant_emotion、dominant_share、vote_ratios、probability_ratios

## 三頁流程

目前流程為：

1. home.html：封面首頁，可播放 white-noise.mp3 白噪音，並記住開關偏好
2. index.html：5 秒情緒辨識主頁，完成後會自動前往 feedback.html
3. feedback.html：收集準確度、滿意度、文字意見，提交到後端統計

## PHP + Spotify 網頁

1. 先啟動情緒辨識程式（讓 JSON 持續更新）
2. 設定 spotify_config.php 內的 client_id / client_secret
3. 雙擊 run_web.bat 或執行下列指令啟動網站：

```powershell
& "C:/xampp/php/php.exe" -S 127.0.0.1:8080
```

4. 瀏覽器打開 http://127.0.0.1:8080

## 注意事項

- 第一次執行會自動下載 FER+ 模型到 models/emotion-ferplus-8.onnx
- 建議單人入鏡、正面面對鏡頭、光線穩定
- 預設已調成較靈敏的人臉偵測；若距離鏡頭較遠可把 --min-face 再調小
- 目前有加入時間平滑與低信心保守輸出，低於門檻會顯示 uncertain
- PHP 網頁的 Spotify 推薦使用多語系、多曲風 query，避免只偏英文流行歌
- 這是專題原型，適合做即時情緒傾向判斷，不適合拿來當心理診斷工具

## AI 桌寵（陰晴）

網頁右下角提供一隻名為「陰晴」的 AI 陪伴桌寵，由 Groq LLM 驅動，可在情緒辨識完成後展開連續對話。

### 設計邊界

- 陰晴是**情緒緩衝夥伴**，不是心理諮商師，也不是診斷工具。
- 不做任何診斷、不給醫療建議、不預測使用者的心理狀態。
- 若對話涉及危機或自傷，陰晴會溫和建議尋求專業協助。

### 人格模式（persona）

`/generate` 支援以下人格模式：

- `assistant`：預設助理，重點回答 + 可執行下一步。
- `courage_coach`：勇氣同理教練模式，採用同理、脆弱、界線與小步行動原則（不模仿特定人物文風）。
- `companion`：陪伴夥伴，回覆更短、更溫柔。

目前 `scan.html` 的桌寵預設採用 `courage_coach`，可先作為「AI 助手人格 MVP」。

### 多聊天室（Gem 風格）

桌寵已支援「多個可保存聊天室」：

- API：`/chat/sessions`、`/chat/sessions/<id>`、`/chat/messages`
- 前端：可新增、切換、刪除聊天室
- 儲存位置：Supabase（不使用後端記憶體保存對話）

這樣可避免 Flask 進程記憶體被長對話塞滿，也能跨裝置保留紀錄。

#### 建議 Supabase 資料表（SQL）

```sql
create table if not exists public.chat_sessions (
	id uuid primary key default gen_random_uuid(),
	user_id uuid not null,
	title text not null default '新的對話',
	persona text not null default 'courage_coach',
	created_at timestamptz not null default now(),
	updated_at timestamptz not null default now(),
	last_message_at timestamptz not null default now()
);

create index if not exists chat_sessions_user_updated_idx
	on public.chat_sessions (user_id, updated_at desc, created_at desc);

create table if not exists public.chat_messages (
	id uuid primary key default gen_random_uuid(),
	user_id uuid not null,
	session_id uuid not null references public.chat_sessions(id) on delete cascade,
	role text not null check (role in ('user', 'assistant')),
	content text not null,
	emotion text not null default 'unknown',
	created_at timestamptz not null default now()
);

create index if not exists chat_messages_session_created_idx
	on public.chat_messages (user_id, session_id, created_at asc, id asc);

-- Agent 長期記憶
create table if not exists public.agent_memories (
	id uuid primary key default gen_random_uuid(),
	user_id uuid not null,
	kind text not null default 'insight',
	content text not null,
	importance int not null default 50,
	tags jsonb not null default '[]'::jsonb,
	source text not null default 'chat',
	session_id uuid,
	created_at timestamptz not null default now(),
	updated_at timestamptz not null default now()
);

create index if not exists agent_memories_user_updated_idx
	on public.agent_memories (user_id, updated_at desc, created_at desc);

-- Agent 任務追蹤
create table if not exists public.agent_tasks (
	id uuid primary key default gen_random_uuid(),
	user_id uuid not null,
	title text not null,
	details text not null default '',
	status text not null default 'open',
	priority text not null default 'normal',
	due_at timestamptz,
	done_at timestamptz,
	source text not null default 'agent',
	created_at timestamptz not null default now(),
	updated_at timestamptz not null default now(),
	constraint agent_tasks_status_check check (status in ('open', 'in_progress', 'done', 'cancelled')),
	constraint agent_tasks_priority_check check (priority in ('low', 'normal', 'high'))
);

create index if not exists agent_tasks_user_updated_idx
	on public.agent_tasks (user_id, updated_at desc, created_at desc);

-- Agent 工具呼叫紀錄
create table if not exists public.agent_tool_logs (
	id uuid primary key default gen_random_uuid(),
	user_id uuid not null,
	tool_name text not null,
	status text not null default 'success',
	input text not null default '',
	output text not null default '',
	error text not null default '',
	latency_ms int not null default 0,
	created_at timestamptz not null default now(),
	constraint agent_tool_logs_status_check check (status in ('success', 'error', 'skipped'))
);

create index if not exists agent_tool_logs_user_created_idx
	on public.agent_tool_logs (user_id, created_at desc, id desc);
```

#### 容量限制（可透過環境變數調整）

- `CHAT_MAX_SESSIONS_PER_USER`：每位使用者最多保留聊天室數量（預設 60）
- `CHAT_MAX_MESSAGES_PER_SESSION`：每個聊天室最多保留訊息數量（預設 300）
- `AGENT_MAX_MEMORIES_PER_USER`：每位使用者最多保留長期記憶數量（預設 300）
- `AGENT_MAX_TOOL_LOGS_PER_USER`：每位使用者最多保留工具紀錄數量（預設 1000）

超出上限時，後端會自動清理最舊資料。

### Agent Step 1 + Step 2 API（已實作）

- `GET /agent/spec`：取得 Agent 的任務範圍與邊界定義
- `GET /agent/memories`：查詢長期記憶
- `POST /agent/memories`：新增長期記憶
- `DELETE /agent/memories/<id>`：刪除長期記憶
- `GET /agent/tasks`：查詢任務
- `POST /agent/tasks`：建立任務
- `PATCH /agent/tasks/<id>`：更新任務狀態/優先級/內容
- `POST /agent/tool-logs`：記錄工具呼叫
- `GET /agent/tool-logs`：讀取工具呼叫紀錄
- `GET /agent/prompt-version`：查看目前 prompt 版本
- `POST /agent/prompt-version`：切換 prompt 版本（需 `X-Agent-Admin-Token`）
- `GET /agent/daily-review`：產出每日回顧摘要
- `POST /agent/followups/next-day`：建立隔天追蹤任務（可被排程器呼叫）

### Agent Step 7（可觀測 + 回歸測試）

`/generate` 現在每一回合都會回傳 `observability`，包含：

- `used_memories` / `used_memory_count`
- `used_tools` / `used_tool_count`
- `latency_ms`
- `fallback_used` / `fallback_reason`
- `safety_mode` / `crisis_level` / `crisis_phase`
- `prompt_version`

若使用者已登入（有 bearer token），後端也會把每回合摘要寫入 `agent_tool_logs`：

- `tool_name = turn_observability`

可直接用 `GET /agent/tool-logs` 檢視每回合品質趨勢。

#### 固定 20 條回歸測試

- 測試案例檔：`regression/agent_regression_cases.json`
- 執行腳本：`regression/run_regression.py`

執行方式：

```powershell
# 選填：測已登入路徑（工具呼叫/記憶/任務）
$env:AGENT_BEARER_TOKEN = "<your_token>"

# 選填：預設是 http://127.0.0.1:8000
$env:AGENT_BASE_URL = "http://127.0.0.1:8000"

& ".venv/Scripts/python.exe" regression/run_regression.py
```

### Agent Step 8（產品化基線）

#### 1) 每日回顧（可由排程器呼叫）

`GET /agent/daily-review?days=1`

- 從近期情緒、未完成任務、最近記憶彙整回顧
- 回傳重點統計與 `recommended_step`

#### 2) 隔天追蹤任務（可由排程器呼叫）

`POST /agent/followups/next-day`

- 會建立「隔天追蹤」任務（避免重複建立）
- 可由外部 scheduler（例如 cron、Render Cron Job）每天觸發

#### 3) Prompt 版本化與回滾

- 讀取目前版本：`GET /agent/prompt-version`
- 切換版本：`POST /agent/prompt-version` with `{ "version": "v1|v2" }`

切換時需帶 header：

```text
X-Agent-Admin-Token: <AGENT_ADMIN_TOKEN>
```

建議在 Render 環境變數設定：

- `AGENT_PROMPT_VERSION=v1`
- `AGENT_ADMIN_TOKEN=<strong-random-token>`

### 啟用方式

1. 在 Render Dashboard 的環境變數設定 `GROQ_API_KEY=你的金鑰`（金鑰**不可**放入程式碼或 git）。
2. 部署後重啟 web service，後端即可處理 `/generate` 請求。

### 安全閥

| 機制 | 設定值 |
|------|--------|
| 每 IP 每小時請求上限 | 30 次 |
| 每輪回覆 token 上限 | 120 tokens |
| API 呼叫超時 | 4 秒（逾時回退固定文案） |
| 單條輸入長度上限 | 500 字元 |
| 對話歷史保留輪數 | 最近 10 輪（20 條） |
| 情緒標籤白名單 | happiness / sadness / anger / disgust / fear / contempt / uncertain / neutral / no_face / unknown |

### 隱私

- 桌寵對話**不儲存於伺服器**，僅在前端當次會話中保留歷史。
- 關閉或重新整理頁面後對話記錄即消失。

## 回饋統計

- 前端會把本次掃描摘要暫存在 sessionStorage，並在 feedback.html 顯示。
- 回饋 API 為 `/feedback`，由後端代理轉送到 Google Sheet。
- 建議在 Render 環境變數設定 `FEEDBACK_WEBHOOK_URL`，值填入 Google Apps Script Web App 的 webhook URL。
- 若 Google Sheet 當下不可寫入，後端會先把資料暫存在 emotion_output/pending_feedback.jsonl，之後有新回饋進來時會自動重送。
- 若使用者送出當下網路中斷，前端也會先暫存在 localStorage，重新打開 feedback.html 時會自動重送。

### Google Sheet 設定方式

1. 建立一份新的 Google Sheet。
2. 打開「擴充功能 → Apps Script」。
3. 把專案中的 [feedback_webhook.gs](feedback_webhook.gs) 內容貼進去並儲存。
4. 在 Apps Script 右上角選「部署 → 新增部署」。
5. 類型選「網頁應用程式」。
6. Execute as 選「Me」，Who has access 選「Anyone」。
7. 部署後複製 Web App URL。
8. 到 Render 的環境變數新增 `FEEDBACK_WEBHOOK_URL=你的 Web App URL`。

### Google Sheet 欄位

- server_received_at
- client_received_at
- source
- ip_hint
- accuracy
- satisfaction
- comment
- emotion
- share
- scan_timestamp

## 白噪音音檔

- 首頁預設讀取專案根目錄的 white-noise.mp3。
- 如果目前還沒放入音檔，首頁仍可正常進入，只是白噪音會顯示未就緒。
