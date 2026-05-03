<?php

function h(string $text): string {
  return htmlspecialchars($text, ENT_QUOTES, 'UTF-8');
}

function detect_python_executable(): ?string
{
  $candidates = [
    __DIR__ . '/.venv/bin/python',
    __DIR__ . '/.venv/Scripts/python.exe',
    'python3',
    'python',
  ];

  foreach ($candidates as $candidate) {
    if ($candidate === 'python' || $candidate === 'python3') {
      @exec($candidate . ' --version 2>&1', $out, $code);
      if ($code === 0) {
        return $candidate;
      }
      continue;
    }
    if (file_exists($candidate)) {
      return $candidate;
    }
  }
  return null;
}

function run_one_shot_capture(int $seconds = 5): array
{
  $python = detect_python_executable();
  if ($python === null) {
    return ['ok' => false, 'message' => '找不到 Python 執行檔'];
  }

  $script = __DIR__ . '/emotion_capture_once.py';
  if (!file_exists($script)) {
    return ['ok' => false, 'message' => '找不到 emotion_capture_once.py'];
  }

  $cmd = escapeshellarg($python)
    . ' ' . escapeshellarg($script)
    . ' --seconds ' . max(5, $seconds)
    . ' 2>&1';

  $output = [];
  $exitCode = 1;
  @exec($cmd, $output, $exitCode);

  return [
    'ok' => ($exitCode === 0),
    'message' => implode("\n", $output),
    'exit_code' => $exitCode,
  ];
}

function reset_emotion_json(string $emotionFile, int $windowSeconds = 5): void
{
  $labels = ['happiness', 'sadness', 'anger', 'disgust', 'fear', 'contempt'];
  $zeros = [];
  foreach ($labels as $label) {
    $zeros[$label] = 0.0;
  }

  $payload = [
    'timestamp' => time(),
    'window_seconds' => $windowSeconds,
    'dominant_emotion' => 'unknown',
    'dominant_share' => 0.0,
    'sample_count' => 0,
    'probability_ratios' => $zeros,
    'vote_ratios' => $zeros,
  ];

  $dir = dirname($emotionFile);
  if (!is_dir($dir)) {
    @mkdir($dir, 0777, true);
  }
  @file_put_contents($emotionFile, json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
}

$emotionFile = __DIR__ . '/emotion_output/latest_emotion.json';

$captureStatus = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'capture_now') {
  $captureStatus = run_one_shot_capture(5);
  $okFlag = !empty($captureStatus['ok']) ? '1' : '0';
  header('Location: ' . $_SERVER['PHP_SELF'] . '?capture_done=1&capture_ok=' . $okFlag);
  exit;
}

$fromCapture = (($_GET['capture_done'] ?? '') === '1');
if (!$fromCapture) {
  // Normal refresh/open page clears previous values.
  reset_emotion_json($emotionFile, 5);
} else {
  $captureStatus = [
    'ok' => (($_GET['capture_ok'] ?? '0') === '1'),
    'message' => '偵測已完成。',
  ];
}
$emotionData = [
    'dominant_emotion' => 'unknown',
    'dominant_share' => 0,
  'window_seconds' => 5,
    'vote_ratios' => [],
    'sample_count' => 0,
];
if (file_exists($emotionFile)) {
    $decoded = json_decode((string) file_get_contents($emotionFile), true);
    if (is_array($decoded)) {
        $emotionData = array_merge($emotionData, $decoded);
    }
}

$emotion = (string) ($emotionData['dominant_emotion'] ?? 'unknown');
$sharePercent = ((float) ($emotionData['dominant_share'] ?? 0)) * 100.0;

// 根據情緒的簡單搜尋關鍵詞 (無需 API)
$emotionQueries = [
    'happiness' => ['happy upbeat pop', 'k-pop', 'mandopop', 'afrobeats'],
    'sadness' => ['sad ballad', 'lofi hip hop', 'indie folk', 'classical piano'],
    'anger' => ['rock metal', 'hip hop rap', 'punk rock', 'heavy music'],
    'disgust' => ['grunge alternative', 'industrial metal', 'post-punk', 'experimental'],
    'fear' => ['ambient dark', 'post-rock', 'neoclassical', 'trip hop'],
    'contempt' => ['jazz fusion', 'neo soul', 'indie french', 'sophisticated'],
    'neutral' => ['lo-fi chill', 'instrumental', 'ambient relaxing', 'background music'],
    'unknown' => ['world music', 'jazz', 'indie'], 
];

$queries = $emotionQueries[$emotion] ?? $emotionQueries['unknown'];
?>
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emotion x Spotify 推薦</title>
  <style>
    :root {
      --bg1: #f2efe7;
      --bg2: #dceefb;
      --ink: #1a1c22;
      --card: #ffffffd8;
      --accent: #1db954;
      --muted: #4b5563;
    }
    body {
      margin: 0;
      font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 500px at 10% -10%, #fff9cf 0%, transparent 60%),
                  radial-gradient(900px 500px at 90% 0%, #d6f6ff 0%, transparent 60%),
                  linear-gradient(145deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }
    .wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px 30px; }
    .hero {
      background: var(--card);
      border: 1px solid #ffffff;
      backdrop-filter: blur(8px);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 30px #00000014;
    }
    .hero h1 { margin: 0 0 8px; font-size: 26px; }
    .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
    .badge {
      border-radius: 999px;
      background: #0f172a;
      color: #fff;
      padding: 6px 12px;
      font-weight: 700;
      font-size: 14px;
    }
    .badge.green { background: var(--accent); color: #062a11; }
    .small { color: var(--muted); font-size: 14px; }
    .grid { margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
    .card {
      background: #fff;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid #eef2f7;
      box-shadow: 0 6px 18px #00000010;
    }
    .cover {
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 10px;
      object-fit: cover;
      background: #f3f4f6;
    }
    .title { font-weight: 700; margin: 10px 0 4px; }
    .artist { color: var(--muted); margin: 0 0 8px; font-size: 14px; }
    .btn {
      display: inline-block;
      text-decoration: none;
      background: #0f172a;
      color: #fff;
      padding: 8px 12px;
      border-radius: 10px;
      font-size: 14px;
      margin-right: 8px;
      margin-top: 4px;
    }
    .btn.spotify { background: var(--accent); color: #05210d; font-weight: 700; }
    .seed-list { margin-top: 10px; color: var(--muted); font-size: 13px; }
    .notice {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 14px;
      background: #f5f8ff;
      border: 1px solid #d7e2ff;
      color: #23314f;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>5 秒情緒回傳 + Spotify 音樂推薦</h1>
      <div class="row">
        <form method="post" style="margin:0;">
          <input type="hidden" name="action" value="capture_now">
          <button class="btn spotify" type="submit">開始 5 秒表情偵測並自動推薦</button>
        </form>
        <span class="badge">主情緒：<?= h($emotion) ?></span>
        <span class="badge green">佔比：<?= number_format($sharePercent, 1) ?>%</span>
        <span class="small">樣本數：<?= (int) ($emotionData['sample_count'] ?? 0) ?></span>
        <span class="small">視窗：<?= (int) ($emotionData['window_seconds'] ?? 15) ?> 秒</span>
      </div>
      <div class="seed-list">推薦音樂類型：<?= h(implode(' | ', $queries)) ?></div>
      <div class="small" style="margin-top:8px;">按下按鈕後，會顯示攝影機預覽並監測 5 秒，完成後自動關閉並把結果回傳到此頁。</div>
      <?php if (is_array($captureStatus)): ?>
        <div class="notice">
          <?= $captureStatus['ok'] ? '偵測完成，已更新推薦。' : '偵測失敗，請檢查攝影機/環境。' ?>
          <?= "\n" . h((string) ($captureStatus['message'] ?? '')) ?>
        </div>
      <?php endif; ?>
    </section>

    <section class="grid" id="spotify-grid">
      <?php foreach ($queries as $q): ?>
        <article class="card">
          <p class="title">🎵 <?= h($q) ?></p>
          <p class="artist">點擊下方按鈕，用 Spotify Web Player 無需登入即可播放</p>
          <a class="btn spotify" target="_blank" rel="noopener" href="https://open.spotify.com/search/<?= rawurlencode($q) ?>">
            🔍 在 Spotify 上搜尋
          </a>
          <br>
          <a class="btn" target="_blank" rel="noopener" href="spotify:search:<?= rawurlencode($q) ?>" style="background: #1DB954; color: white;">
            🎧 用 Spotify App 開啟
          </a>
        </article>
      <?php endforeach; ?>
    </section>
    
    <section class="hero" style="margin-top: 30px;">
      <h2>📱 手機攝像頭實時檢測</h2>
      <div style="margin-top: 12px;">
        <button id="start-camera" class="btn spotify" type="button">啟動手機攝像頭</button>
        <button id="stop-camera" class="btn" type="button" style="display:none; background: #dc2626;">關閉攝像頭</button>
      </div>
      <div id="camera-status" style="margin-top: 12px; font-size: 14px; color: #666;"></div>
      <canvas id="camera-canvas" style="max-width: 100%; margin-top: 12px; border-radius: 10px; display: none;"></canvas>
      <div id="detection-result" style="margin-top: 12px; padding: 12px; background: #f0fdf4; border-radius: 10px; display: none;">
        <p><strong>檢測結果：</strong> <span id="result-emotion">-</span></p>
        <p><strong>信心度：</strong> <span id="result-confidence">-</span></p>
      </div>
    </section>
  </div>
</body>
<script>
const startBtn = document.getElementById('start-camera');
const stopBtn = document.getElementById('stop-camera');
const statusDiv = document.getElementById('camera-status');
const canvas = document.getElementById('camera-canvas');
const resultDiv = document.getElementById('detection-result');
const emotionSpan = document.getElementById('result-emotion');
const confidenceSpan = document.getElementById('result-confidence');

let stream = null;
let video = null;
let detectionInterval = null;

startBtn.addEventListener('click', async () => {
  try {
    statusDiv.textContent = '📍 申請攝像頭權限...';
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    
    video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    
    canvas.style.display = 'block';
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-block';
    statusDiv.textContent = '✅ 攝像頭已啟動，開始檢測...';
    resultDiv.style.display = 'block';
    
    // 每 500ms 檢測一次
    detectionInterval = setInterval(async () => {
      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob(async (blob) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
          try {
            const response = await fetch('camera.php', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ frame: e.target.result }),
            });
            
            if (response.ok) {
              const data = await response.json();
              if (data.dominant_emotion && data.dominant_emotion !== 'no_face') {
                emotionSpan.textContent = data.dominant_emotion.toUpperCase();
                confidenceSpan.textContent = (data.confidence * 100).toFixed(1) + '%';
              } else {
                emotionSpan.textContent = '偵測中...';
                confidenceSpan.textContent = '-';
              }
            }
          } catch (err) {
            statusDiv.textContent = '❌ 檢測失敗：' + err.message;
          }
        };
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.8);
    }, 500);
    
  } catch (err) {
    statusDiv.textContent = '❌ 攝像頭存取失敗：' + err.message;
  }
});

stopBtn.addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (detectionInterval) {
    clearInterval(detectionInterval);
    detectionInterval = null;
  }
  canvas.style.display = 'none';
  resultDiv.style.display = 'none';
  startBtn.style.display = 'inline-block';
  stopBtn.style.display = 'none';
  statusDiv.textContent = '';
});
</script>
<?php if ($fromCapture): ?>
<script>
  // Remove query params so next F5 is a normal GET refresh.
  if (window.history && window.history.replaceState) {
    window.history.replaceState({}, document.title, <?= json_encode($_SERVER['PHP_SELF']) ?>);
  }
</script>
<?php endif; ?>
</html>
