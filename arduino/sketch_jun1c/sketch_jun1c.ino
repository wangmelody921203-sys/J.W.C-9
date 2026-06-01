// ==========================================
// IPC Team 專題前端：定時定量平均值輸出系統（含範圍自動放大版）
// ==========================================

const int fsrPin = A0;      // 壓力感測器接在類比腳位 A0
const int ledPin = 3;       // 單色 LED 接在數位腳位 3 (支援 PWM)

// --- 感測降敏參數（可依手感微調）---
const int DEADZONE = 22;          // 小於此範圍的變化視為雜訊
const float RESPONSE_GAMMA = 1.9; // >1 越大代表越不敏感（更不易衝高）
const float LED_SMOOTH_ALPHA = 0.20;

// --- 時間窗口控制參數 ---
unsigned long lastLogTime = 0;     // 記錄上一次發送數據的時間
const unsigned long interval = 1000; // 設定每隔多久要算一次平均傳給電腦（1秒）

// --- 取樣累加變數 ---
long sampleTotal = 0;       // 用來累加這 1 秒內所有讀取到的數值
int sampleCount = 0;        // 記錄這 1 秒內到底讀取了幾次

int baseline = 0;           // 開機時自動校正的基準值
float ledFiltered = 0.0;    // LED 顯示用 EMA 濾波

int convertRawToScaled(int raw) {
  int adjusted = raw - baseline - DEADZONE;
  if (adjusted < 0) {
    adjusted = 0;
  }

  int dynamicMax = 1023 - baseline - DEADZONE;
  if (dynamicMax < 1) {
    dynamicMax = 1;
  }

  float normalized = (float)adjusted / (float)dynamicMax;
  normalized = constrain(normalized, 0.0, 1.0);

  // 非線性壓縮：小力更平緩，大力才會接近滿分
  float curved = pow(normalized, RESPONSE_GAMMA);
  int scaled = (int)(curved * 1023.0 + 0.5);
  return constrain(scaled, 0, 1023);
}

void setup() {
  Serial.begin(9600);      // 啟動序列埠通訊
  pinMode(ledPin, OUTPUT); // LED 為輸出模式

  // 開機前 1 秒抓環境基準值，避免每塊板子的零點偏移造成過敏
  long sum = 0;
  const int calibrateCount = 220;
  for (int i = 0; i < calibrateCount; i++) {
    sum += analogRead(fsrPin);
    delay(4);
  }
  baseline = (int)(sum / calibrateCount);
}

void loop() {
  // 【步驟 1】無論何時，Arduino 都在全速讀取壓力並提供即時的 LED 視覺回饋
  int rawValue = analogRead(fsrPin);

  int scaledInstant = convertRawToScaled(rawValue);

  // LED 走低通濾波，避免一點點抖動就閃很大
  ledFiltered = (1.0 - LED_SMOOTH_ALPHA) * ledFiltered + LED_SMOOTH_ALPHA * (float)scaledInstant;
  int ledBrightness = map((int)ledFiltered, 0, 1023, 0, 255);
  ledBrightness = constrain(ledBrightness, 0, 255);
  analogWrite(ledPin, ledBrightness);

  // 【步驟 2】將這一瞬間的原始數值，累加到我們的时间窗口暫存器裡
  sampleTotal += rawValue;
  sampleCount++;

  // 【步驟 3】檢查 1 秒鐘的時間到了沒？
  unsigned long currentMillis = millis(); 
  
  if (currentMillis - lastLogTime >= interval) {
    if (sampleCount <= 0) {
      sampleCount = 1;
    }

    // 1. 計算 1 秒內的平均值，再做降敏轉換
    int finalAverage = sampleTotal / sampleCount;
    int scaledValue = convertRawToScaled(finalAverage);
    
    // 2. 【丟數據】將這個被軟體放大、完美的數字噴給 Vibe Coding
    Serial.println(scaledValue);
    
    // 3. 重置計時器與暫存器，準備計算下一個 1 秒鐘
    sampleTotal = 0;
    sampleCount = 0;
    lastLogTime = currentMillis; 
  }

  delay(5); 
}
