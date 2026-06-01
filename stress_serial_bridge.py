from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import serial
    from serial import SerialException
except Exception as exc:  # pragma: no cover
    print("Missing dependency: pyserial. Install with `pip install pyserial`.", file=sys.stderr)
    raise


DEFAULT_OUTPUT = Path("emotion_output/latest_stress.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Arduino FSR values from serial and export normalized stress JSON.",
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM4")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON file path")
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=5,
        help="Mark payload stale if no new sample within this many seconds",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="EMA smoothing factor for sensor value (0.0~1.0)",
    )
    parser.add_argument(
        "--push-url",
        default="",
        help="Optional cloud endpoint URL to push stress payloads, e.g. https://<host>/stress/report",
    )
    parser.add_argument(
        "--push-token",
        default="",
        help="Optional token sent in X-Stress-Token header when --push-url is used",
    )
    parser.add_argument(
        "--user-id",
        default="",
        help="Optional user_id to tag payload for per-user stress stream",
    )
    return parser.parse_args()


def classify_stress_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def to_payload(sensor_value: int, smoothed_value: float, stale_seconds: int) -> dict:
    clipped = max(0, min(1023, int(sensor_value)))
    normalized = clipped / 1023.0
    score = int(round(normalized * 100))

    return {
        "source": "serial_fsr",
        "timestamp": int(time.time()),
        "is_stale": False,
        "stale_seconds": max(1, int(stale_seconds)),
        "sensor_value": clipped,
        "smoothed_value": round(float(smoothed_value), 2),
        "normalized": round(float(normalized), 4),
        "stress_score": score,
        "stress_level": classify_stress_level(score),
    }


def parse_int_from_line(raw: bytes) -> int | None:
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group(0))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stale(path: Path, stale_seconds: int) -> None:
    payload = {
        "source": "serial_fsr",
        "timestamp": int(time.time()),
        "is_stale": True,
        "stale_seconds": max(1, int(stale_seconds)),
        "sensor_value": 0,
        "smoothed_value": 0.0,
        "normalized": 0.0,
        "stress_score": 0,
        "stress_level": "unknown",
    }
    write_json(path, payload)


def push_payload(push_url: str, payload: dict, push_token: str = "") -> bool:
    if not push_url:
        return False

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if push_token:
        headers["X-Stress-Token"] = push_token

    req = urllib.request.Request(push_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def run_loop(args: argparse.Namespace) -> int:
    alpha = float(max(0.0, min(1.0, args.alpha)))
    smoothed: float | None = None
    last_sample_at = 0.0
    push_url = str(args.push_url or "").strip()
    push_token = str(args.push_token or "").strip()
    user_id = str(args.user_id or "").strip()

    print(f"Opening serial port {args.port} @ {args.baud}...")
    if push_url:
        print(f"Cloud push enabled: {push_url}")
    if user_id:
        print(f"Per-user mode enabled for user_id: {user_id}")
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            time.sleep(1.5)
            print("Stress bridge started. Press Ctrl+C to stop.")
            while True:
                line = ser.readline()
                value = parse_int_from_line(line)
                now = time.time()

                if value is None:
                    if last_sample_at > 0 and (now - last_sample_at) > max(1, int(args.stale_seconds)):
                        write_stale(args.output, args.stale_seconds)
                    continue

                if smoothed is None:
                    smoothed = float(value)
                else:
                    smoothed = (1.0 - alpha) * smoothed + alpha * float(value)

                last_sample_at = now
                payload = to_payload(value, smoothed, args.stale_seconds)
                if user_id:
                    payload["user_id"] = user_id
                write_json(args.output, payload)
                pushed = push_payload(push_url, payload, push_token) if push_url else False
                push_text = " pushed=ok" if pushed else (" pushed=fail" if push_url else "")
                print(
                    f"raw={payload['sensor_value']:4d} score={payload['stress_score']:3d} level={payload['stress_level']}{push_text}"
                )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_loop(arguments))
