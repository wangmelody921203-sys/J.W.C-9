from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
MODEL_PATH = Path("models/emotion-ferplus-8.onnx")
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
EMOTION_LABELS = [
    "happiness",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# FER+ outputs 8 classes; app-level outputs exclude surprise (2) and neutral (0).
ACTIVE_CLASS_INDICES = [1, 3, 4, 5, 6, 7]
WINDOW_SECONDS = 5
OUTPUT_FILE = Path("emotion_output/latest_emotion.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV facial emotion recognition prototype")
    parser.add_argument("--camera", type=int, default=0, help="Camera index passed to OpenCV")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the FER+ ONNX model file",
    )
    parser.add_argument(
        "--min-face",
        type=int,
        default=48,
        help="Minimum detected face size in pixels",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.18,
        help="Extra crop padding around detected face box (0.0~0.5)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.28,
        help="EMA smoothing factor for emotion probabilities (0.0~1.0)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Below this confidence, show uncertain instead of hard label",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Disable horizontal mirror preview",
    )
    parser.set_defaults(mirror=True)
    parser.add_argument(
        "--neutral-penalty",
        type=float,
        default=0.5,
        help="Subtract this from neutral's raw logit before softmax to reduce over-prediction of neutral (0.0~2.0)",
    )
    parser.add_argument(
        "--emotion-boost",
        type=float,
        default=1.3,
        help="Multiply all non-neutral class logits by this factor before softmax (1.0~2.5)",
    )
    parser.add_argument(
        "--neutral-cap",
        type=float,
        default=0.40,
        help="Cap neutral probability after softmax to avoid neutral over-dominance (0.1~0.9)",
    )
    parser.add_argument(
        "--expressive-margin",
        type=float,
        default=0.14,
        help="If top1-top2 probability gap is below this margin, classify as uncertain",
    )
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    if model_path.exists() and is_valid_model_file(model_path):
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FER+ model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)

    if not is_valid_model_file(model_path):
        raise RuntimeError("Downloaded model is not a valid ONNX file.")

    return model_path


def is_valid_model_file(model_path: Path) -> bool:
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        return False

    header = model_path.read_bytes()[:64]
    return not header.startswith(b"version https://git-lfs.github.com/spec/v1")


def get_asset_cache_dir() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "opencv_emotion_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_asset(source_path: Path) -> Path:
    cached_path = get_asset_cache_dir() / source_path.name
    if not cached_path.exists() or cached_path.stat().st_size != source_path.stat().st_size:
        cached_path.write_bytes(source_path.read_bytes())
    return cached_path


def load_face_detector() -> cv2.CascadeClassifier:
    source_path = Path(cv2.data.haarcascades) / CASCADE_FILENAME
    cascade_path = cache_asset(source_path)
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return detector


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    enhanced = cv2.equalizeHist(gray_face)
    resized = cv2.resize(enhanced, (64, 64), interpolation=cv2.INTER_AREA)
    input_tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
    return input_tensor


def softmax(scores: np.ndarray) -> np.ndarray:
    stabilized = scores - np.max(scores)
    exp_scores = np.exp(stabilized)
    return exp_scores / np.sum(exp_scores)


def load_emotion_session(model_path: Path) -> ort.InferenceSession:
    runtime_model_path = cache_asset(model_path)
    return ort.InferenceSession(str(runtime_model_path), providers=["CPUExecutionProvider"])


def classify_emotion(
    session: ort.InferenceSession,
    gray_face: np.ndarray,
    neutral_penalty: float = 0.0,
    emotion_boost: float = 1.0,
) -> tuple[str, np.ndarray]:
    input_tensor = preprocess_face(gray_face)
    input_name = session.get_inputs()[0].name
    scores = session.run(None, {input_name: input_tensor})[0].flatten()[ACTIVE_CLASS_INDICES].astype(np.float64)

    # Output now contains expressive classes only (no neutral).
    scores *= float(np.clip(emotion_boost, 1.0, 5.0))

    probabilities = softmax(scores)
    best_index = int(np.argmax(probabilities))
    return EMOTION_LABELS[best_index], probabilities


def rebalance_probabilities(probabilities: np.ndarray, neutral_cap: float) -> np.ndarray:
    # No neutral class in output labels; keep probabilities normalized and unchanged.
    adjusted = probabilities.astype(np.float64).copy()
    adjusted /= max(np.sum(adjusted), 1e-9)
    return adjusted


def resolve_emotion_label(
    probabilities: np.ndarray,
    confidence_threshold: float,
    expressive_margin: float,
    happy_sad_confidence_bonus: float = 0.08,
    happy_sad_margin_bonus: float = 0.04,
) -> tuple[str, int, float, bool]:
    ranked = np.argsort(probabilities)[::-1]
    best_index = int(ranked[0])
    second_index = int(ranked[1])

    confidence = float(probabilities[best_index])
    gap = float(probabilities[best_index] - probabilities[second_index])
    best_label = EMOTION_LABELS[best_index]

    required_confidence = confidence_threshold
    required_margin = expressive_margin
    if best_label in ("happiness", "sadness"):
        required_confidence += max(0.0, happy_sad_confidence_bonus)
        required_margin += max(0.0, happy_sad_margin_bonus)

    if confidence < required_confidence or gap < required_margin:
        return f"uncertain ({confidence * 100:.1f}%)", best_index, confidence, False
    return f"{EMOTION_LABELS[best_index]} ({confidence * 100:.1f}%)", best_index, confidence, True


def draw_probability_panel(frame: np.ndarray, probabilities: np.ndarray) -> None:
    panel_x = 12
    panel_y = 40
    row_height = 24
    bar_width = 220

    cv2.putText(
        frame,
        "Emotion probabilities",
        (panel_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, label in enumerate(EMOTION_LABELS):
        probability = float(probabilities[index])
        y = panel_y + index * row_height
        top_left = (panel_x + 130, y - 12)
        bottom_right = (panel_x + 130 + bar_width, y + 4)
        fill_right = (panel_x + 130 + int(bar_width * probability), y + 4)
        cv2.putText(
            frame,
            f"{label:>10}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, top_left, bottom_right, (80, 80, 80), 1)
        cv2.rectangle(frame, top_left, fill_right, (0, 190, 255), cv2.FILLED)
        cv2.putText(
            frame,
            f"{probability * 100:5.1f}%",
            (panel_x + 360, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Cannot open camera. Check webcam permissions and camera index.")
    return capture


def detect_faces(detector: cv2.CascadeClassifier, gray: np.ndarray, min_face: int) -> np.ndarray:
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(min_face, min_face),
    )
    if len(faces) > 0:
        return faces

    # Fallback pass: equalize image and relax thresholds for dim light / small faces.
    enhanced = cv2.equalizeHist(gray)
    relaxed_min = max(32, min_face // 2)
    return detector.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(relaxed_min, relaxed_min),
    )


def padded_face_region(gray: np.ndarray, face_box: tuple[int, int, int, int], padding: float) -> np.ndarray:
    x, y, w, h = face_box
    pad_ratio = float(np.clip(padding, 0.0, 0.5))
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(gray.shape[1], x + w + pad_w)
    y2 = min(gray.shape[0], y + h + pad_h)
    return gray[y1:y2, x1:x2]


def prune_window(samples: deque[tuple[float, np.ndarray]], now: float, window_seconds: int) -> None:
    while samples and now - samples[0][0] > window_seconds:
        samples.popleft()


def summarize_window(samples: deque[tuple[float, np.ndarray]]) -> tuple[str, float, dict[str, float], dict[str, float], int]:
    if not samples:
        empty = {label: 0.0 for label in EMOTION_LABELS}
        return "unknown", 0.0, empty, empty, 0

    stacked = np.stack([prob for _, prob in samples], axis=0)
    probability_ratios = stacked.mean(axis=0)

    vote_counts = np.zeros(len(EMOTION_LABELS), dtype=np.float64)
    for row in stacked:
        vote_counts[int(np.argmax(row))] += 1.0
    vote_ratios = vote_counts / len(samples)

    best_index = int(np.argmax(vote_ratios))
    dominant = EMOTION_LABELS[best_index]
    dominant_share = float(vote_ratios[best_index])

    prob_map = {label: float(probability_ratios[i]) for i, label in enumerate(EMOTION_LABELS)}
    vote_map = {label: float(vote_ratios[i]) for i, label in enumerate(EMOTION_LABELS)}
    return dominant, dominant_share, prob_map, vote_map, len(samples)


def export_window_result(
    output_file: Path,
    dominant: str,
    dominant_share: float,
    probability_ratios: dict[str, float],
    vote_ratios: dict[str, float],
    sample_count: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "window_seconds": WINDOW_SECONDS,
        "dominant_emotion": dominant,
        "dominant_share": dominant_share,
        "sample_count": sample_count,
        "probability_ratios": probability_ratios,
        "vote_ratios": vote_ratios,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_output_result(output_file: Path, window_seconds: int = WINDOW_SECONDS) -> None:
    empty = {label: 0.0 for label in EMOTION_LABELS}
    payload = {
        "timestamp": int(time.time()),
        "window_seconds": window_seconds,
        "dominant_emotion": "unknown",
        "dominant_share": 0.0,
        "sample_count": 0,
        "probability_ratios": empty,
        "vote_ratios": empty,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        model_path = ensure_model(args.model)
        detector = load_face_detector()
        emotion_session = load_emotion_session(model_path)
        capture = open_camera(args.camera)
        reset_output_result(OUTPUT_FILE, WINDOW_SECONDS)
    except Exception as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1

    print("Press Q to quit. Best results: one person, front-facing, steady lighting.")
    last_probabilities = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    smoothed_probabilities: np.ndarray | None = None
    emotion_window: deque[tuple[float, np.ndarray]] = deque()
    dominant_15s = "unknown"
    dominant_share_15s = 0.0
    last_export_time = 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read a frame from the camera.", file=sys.stderr)
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray, args.min_face)

        status_text = "No face detected"
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face_region = padded_face_region(gray, (x, y, w, h), args.face_padding)
            emotion_label, probabilities = classify_emotion(
                emotion_session,
                face_region,
                neutral_penalty=args.neutral_penalty,
                emotion_boost=args.emotion_boost,
            )

            alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
            if smoothed_probabilities is None:
                smoothed_probabilities = probabilities
            else:
                smoothed_probabilities = (1.0 - alpha) * smoothed_probabilities + alpha * probabilities

            calibrated_probabilities = rebalance_probabilities(smoothed_probabilities, args.neutral_cap)
            last_probabilities = calibrated_probabilities
            status_text, _, _, should_count = resolve_emotion_label(
                calibrated_probabilities,
                args.confidence_threshold,
                args.expressive_margin,
            )

            now = time.time()
            if should_count:
                emotion_window.append((now, calibrated_probabilities.copy()))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 255), 2)
            cv2.putText(
                frame,
                status_text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 190, 255),
                2,
                cv2.LINE_AA,
            )

        now = time.time()
        prune_window(emotion_window, now, WINDOW_SECONDS)
        dominant_15s, dominant_share_15s, probability_ratios, vote_ratios, sample_count = summarize_window(emotion_window)
        if now - last_export_time >= 1.0:
            export_window_result(
                OUTPUT_FILE,
                dominant_15s,
                dominant_share_15s,
                probability_ratios,
                vote_ratios,
                sample_count,
            )
            last_export_time = now

        cv2.putText(
            frame,
            status_text,
            (12, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{WINDOW_SECONDS}s dominant: {dominant_15s} ({dominant_share_15s * 100:.1f}%)",
            (12, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        draw_probability_panel(frame, last_probabilities)
        cv2.imshow("Emotion Recognition Prototype", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())