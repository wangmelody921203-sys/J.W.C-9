// ==========================================
// IPC Team 專題前端：定時定量平均值輸出系統（含範圍自動放大版）
// ==========================================

const int fsrPin = A0;      // 壓力感測器接在類比腳位 A0
const int ledPin = 3;       // 單色 LED 接在數位腳位 3 (支援 PWM)

// --- 時間窗口控制參數 ---
unsigned long lastLogTime = 0;     // 記錄上一次發送數據的時間
const unsigned long interval = 1000; // 設定每隔多久要算一次平均傳給電腦（1秒）

// --- 取樣累加變數 ---
long sampleTotal = 0;       // 用來累加這 1 秒內所有讀取到的數值
int sampleCount = 0;        // 記錄這 1 秒內到底讀取了幾次

void setup() {
  Serial.begin(9600);      // 啟動序列埠通訊
  pinMode(ledPin, OUTPUT); // LED 為輸出模式
}

void loop() {
  // 【步驟 1】無論何時，Arduino 都在全速讀取壓力並提供即時的 LED 視覺回饋
  int rawValue = analogRead(fsrPin);
  
  // LED 依然維持極度即時的微秒級反應
  int ledBrightness = map(rawValue, 560, 1023, 0, 255); // 讓 LED 亮度也適應新電腦的範圍
  ledBrightness = constrain(ledBrightness, 0, 255);
  analogWrite(ledPin, ledBrightness);

  // 【步驟 2】將這一瞬間的原始數值，累加到我們的时间窗口暫存器裡
  sampleTotal += rawValue;
  sampleCount++;

  // 【步驟 3】檢查 1 秒鐘的時間到了沒？
  unsigned long currentMillis = millis(); 
  
  if (currentMillis - lastLogTime >= interval) {
    
    // 1. 計算這 1 秒內的最終原始平均值（此時範圍在 560 ~ 1023 之間）
    int finalAverage = sampleTotal / sampleCount;
    
    // ====================================================
    // :star:【資工黑科技：動態範圍縮放】:star:
    // 把 560~1023 的尷尬範圍，完美映射放大到 0~1023 的標準範圍
    // ====================================================
    int scaledValue = map(finalAverage, 560, 1023, 0, 1023); 
    
    // 防呆機制：限制輸出的數字絕對要在 0 ~ 1023 之間，不可以爆出去
    scaledValue = constrain(scaledValue, 0, 1023); 
    // ====================================================
    
    // 2. 【丟數據】將這個被軟體放大、完美的數字噴給 Vibe Coding
    Serial.println(scaledValue);
    
    // 3. 重置計時器與暫存器，準備計算下一個 1 秒鐘
    sampleTotal = 0;
    sampleCount = 0;
    lastLogTime = currentMillis; 
  }

  delay(5); 
}
