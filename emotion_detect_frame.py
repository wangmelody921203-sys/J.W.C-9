from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from emotion_camera import (
    EMOTION_LABELS,
    load_face_detector,
    load_emotion_session,
    ensure_model,
    classify_emotion,
    rebalance_probabilities,
    resolve_emotion_label,
    detect_faces,
    padded_face_region,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single frame emotion detection")
    parser.add_argument("--frame", type=Path, required=True, help="Path to input frame image")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--expressive-margin", type=float, default=0.14)
    parser.add_argument("--min-face", type=int, default=48)
    parser.add_argument("--face-padding", type=float, default=0.22)
    parser.add_argument("--smooth-alpha", type=float, default=0.22)
    parser.add_argument("--neutral-penalty", type=float, default=0.5)
    parser.add_argument("--emotion-boost", type=float, default=1.3)
    parser.add_argument("--neutral-cap", type=float, default=0.40)
    parser.add_argument("--happy-sad-confidence-bonus", type=float, default=0.08)
    parser.add_argument("--happy-sad-margin-bonus", type=float, default=0.04)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 讀取輸入圖片
    if not args.frame.exists():
        print(json.dumps({"error": "Frame file not found"}))
        return 1

    frame = cv2.imread(str(args.frame))
    if frame is None:
        print(json.dumps({"error": "Failed to read frame"}))
        return 1

    try:
        model_path = ensure_model(Path("models/emotion-ferplus-8.onnx"))
        detector = load_face_detector()
        session = load_emotion_session(model_path)
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        return 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(detector, gray, args.min_face)

    if len(faces) == 0:
        print(json.dumps({
            "dominant_emotion": "no_face",
            "confidence": 0.0,
            "all_probabilities": {label: 0.0 for label in EMOTION_LABELS},
        }))
        return 0

    # 取最大的臉
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_region = padded_face_region(gray, (x, y, w, h), args.face_padding)

    emotion_label, probabilities = classify_emotion(
        session,
        face_region,
        neutral_penalty=args.neutral_penalty,
        emotion_boost=args.emotion_boost,
    )

    calibrated = rebalance_probabilities(probabilities, args.neutral_cap)
    status, best_idx, confidence, should_count = resolve_emotion_label(
        calibrated,
        args.confidence_threshold,
        args.expressive_margin,
    )

    result = {
        "dominant_emotion": EMOTION_LABELS[best_idx] if should_count else "uncertain",
        "confidence": float(confidence),
        "all_probabilities": {label: float(calibrated[i]) for i, label in enumerate(EMOTION_LABELS)},
        "face_detected": True,
    }

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
