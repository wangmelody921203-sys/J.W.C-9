<?php
// 接收手機攝像頭傳來的影像幀，存檔並觸發 Python 檢測

header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

// 接收 Base64 編碼的攝像頭幀
$input = json_decode(file_get_contents('php://input'), true);

if (!isset($input['frame'])) {
    http_response_code(400);
    echo json_encode(['error' => 'No frame data']);
    exit;
}

// 將 Base64 轉為圖檔
$frame_data = str_replace('data:image/jpeg;base64,', '', $input['frame']);
$decoded = base64_decode($frame_data);

if ($decoded === false) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid base64']);
    exit;
}

// 存到臨時檔案
$temp_file = tempnam(sys_get_temp_dir(), 'emotion_frame_') . '.jpg';
file_put_contents($temp_file, $decoded);

// 呼叫 Python 做單幀檢測（新增一個單幀偵測腳本）
$python = detect_python_executable();
if ($python === null) {
    http_response_code(500);
    echo json_encode(['error' => 'Python not found']);
    unlink($temp_file);
    exit;
}

$script = __DIR__ . '/emotion_detect_frame.py';
if (!file_exists($script)) {
    http_response_code(500);
    echo json_encode(['error' => 'Detection script not found']);
    unlink($temp_file);
    exit;
}

$cmd = escapeshellarg($python)
    . ' ' . escapeshellarg($script)
    . ' --frame ' . escapeshellarg($temp_file)
    . ' 2>&1';

$output = [];
$exit_code = 0;
@exec($cmd, $output, $exit_code);

unlink($temp_file);

if ($exit_code === 0 && !empty($output)) {
    $result = json_decode(implode("\n", $output), true);
    if (!is_array($result)) {
        $result = [];
    }
    $result['stress'] = read_latest_stress();
    echo json_encode($result);
} else {
    echo json_encode([
        'error' => 'Detection failed',
        'stress' => read_latest_stress(),
    ]);
}

function read_latest_stress(): array
{
  $default = [
    'source' => 'unavailable',
    'timestamp' => 0,
    'is_stale' => true,
    'sensor_value' => 0,
    'smoothed_value' => 0.0,
    'normalized' => 0.0,
    'stress_score' => 0,
    'stress_level' => 'unknown',
  ];

  $file = __DIR__ . '/emotion_output/latest_stress.json';
  if (!file_exists($file)) {
    return $default;
  }

  $raw = @file_get_contents($file);
  if ($raw === false) {
    return $default;
  }

  $decoded = json_decode($raw, true);
  if (!is_array($decoded)) {
    return $default;
  }

  return array_merge($default, $decoded);
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
?>
