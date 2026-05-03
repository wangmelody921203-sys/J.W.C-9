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
