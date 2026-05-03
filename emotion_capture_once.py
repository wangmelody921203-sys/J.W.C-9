from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from emotion_camera import (
    EMOTION_LABELS,
    OUTPUT_FILE,
    reset_output_result,
    rebalance_probabilities,
    resolve_emotion_label,
    classify_emotion,
    detect_faces,
    ensure_model,
    export_window_result,
    load_emotion_session,
    load_face_detector,
    open_camera,
    padded_face_region,
    prune_window,
    summarize_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless 5s emotion capture")
    parser.add_argument("--seconds", type=int, default=5, help="Capture duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--min-face", type=int, default=48, help="Minimum face size")
    parser.add_argument("--face-padding", type=float, default=0.22, help="Face crop padding ratio")
    parser.add_argument("--smooth-alpha", type=float, default=0.22, help="EMA smoothing alpha")
    parser.add_argument("--neutral-penalty", type=float, default=0.5, help="Neutral class penalty")
    parser.add_argument("--emotion-boost", type=float, default=1.3, help="Expressive class boost")
    parser.add_argument("--neutral-cap", type=float, default=0.40, help="Neutral probability cap")
    parser.add_argument("--expressive-margin", type=float, default=0.14, help="Top1-top2 minimum gap for clear expression")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Uncertain threshold")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds to count down before monitoring starts")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output JSON path")
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Disable horizontal mirror preview",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable camera preview window during capture",
    )
    parser.set_defaults(preview=True, mirror=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reset_output_result(args.output, args.seconds)
    model_path = ensure_model(Path("models/emotion-ferplus-8.onnx"))
    detector = load_face_detector()
    session = load_emotion_session(model_path)
    capture = open_camera(args.camera)

    window_name = "Emotion Capture (Auto Close)"
    if args.preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    countdown = max(0, args.countdown)
    if countdown > 0:
        countdown_deadline = time.time() + countdown
        while time.time() < countdown_deadline:
            ok, frame = capture.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            if args.preview:
                remain = max(0.0, countdown_deadline - time.time())
                cv2.putText(frame, f"Starting in {remain:0.1f}s", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Please face the camera", (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 210, 120), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    capture.release()
                    cv2.destroyAllWindows()
                    return 0
            else:
                time.sleep(0.03)

    smoothed: np.ndarray | None = None
    emotion_window: deque[tuple[float, np.ndarray]] = deque()
    deadline = time.time() + max(5, args.seconds)

    while time.time() < deadline:
        ok, frame = capture.read()
        if not ok:
            continue

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray, args.min_face)
        status = "No face"
        if len(faces) == 0:
            prune_window(emotion_window, time.time(), args.seconds)
            if args.preview:
                remain = max(0.0, deadline - time.time())
                cv2.putText(frame, f"Monitoring... {remain:0.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 210, 120), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
            continue

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_region = padded_face_region(gray, (x, y, w, h), args.face_padding)
        _, probabilities = classify_emotion(
            session,
            face_region,
            neutral_penalty=args.neutral_penalty,
            emotion_boost=args.emotion_boost,
        )

        alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
        if smoothed is None:
            smoothed = probabilities
        else:
            smoothed = (1.0 - alpha) * smoothed + alpha * probabilities

        calibrated = rebalance_probabilities(smoothed, args.neutral_cap)

        status, _, _, should_count = resolve_emotion_label(
            calibrated,
            args.confidence_threshold,
            args.expressive_margin,
        )

        if not should_count:
            if args.preview:
                remain = max(0.0, deadline - time.time())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 255), 2)
                cv2.putText(frame, f"Monitoring... {remain:0.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
            continue

        if args.preview:
            remain = max(0.0, deadline - time.time())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 255), 2)
            cv2.putText(frame, f"Monitoring... {remain:0.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

        now = time.time()
        emotion_window.append((now, calibrated.copy()))
        prune_window(emotion_window, now, args.seconds)

    capture.release()
    if args.preview:
        cv2.destroyAllWindows()

    dominant, dominant_share, prob_map, vote_map, sample_count = summarize_window(emotion_window)
    export_window_result(args.output, dominant, dominant_share, prob_map, vote_map, sample_count)

    print(f"dominant_emotion={dominant}")
    print(f"dominant_share={dominant_share:.4f}")
    print(f"sample_count={sample_count}")
    print(f"labels={','.join(EMOTION_LABELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())