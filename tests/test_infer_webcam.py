from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .test_predict import (
    StreamingPredictor,
    build_inference_objects,
)
from src.utils.overlay import draw_overlay
from src.utils.tts_worker import TTSWorker
from src.utils.webcam import Webcam


DEFAULT_MODEL_PATH = Path("models/checkpoints/asl_gru_best.pt")
DEFAULT_DATA_META = Path("data/processed/train_meta.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live ASL inference demo")

    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--meta", type=Path, default=DEFAULT_DATA_META)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--record-fps", type=float, default=15.0)
    p.add_argument("--min-history", type=float, default=1.0)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument(
        "--silent-when-no-hands",
        action="store_true",
        help="Hide prediction when no hands are detected",
    )

    return p.parse_args()


def compute_hand_motion(prev_hands, curr_hands) -> float:
    """
    Estimate average landmark motion between two consecutive frames.
    Returns 0.0 if motion cannot be computed.
    """
    if not prev_hands or not curr_hands:
        return 0.0

    try:
        prev_landmarks = prev_hands[0]["landmarks"]
        curr_landmarks = curr_hands[0]["landmarks"]
    except Exception:
        return 0.0

    if len(prev_landmarks) != len(curr_landmarks):
        return 0.0

    total = 0.0
    count = 0

    for p, c in zip(prev_landmarks, curr_landmarks):
        px, py = float(p[0]), float(p[1])
        cx, cy = float(c[0]), float(c[1])

        dx = cx - px
        dy = cy - py

        total += (dx * dx + dy * dy) ** 0.5
        count += 1

    if count == 0:
        return 0.0

    return total / count


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.meta.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.meta}")

    model, feature_builder, seq_len, pad_mode, id_to_label = build_inference_objects(
        model_path=args.model,
        meta_path=args.meta,
    )

    from src.utils.hand_detector import HandDetector

    cam = Webcam(camera_index=args.camera_index)
    detector = HandDetector()

    predictor = StreamingPredictor(
        model=model,
        feature_builder=feature_builder,
        seq_len=seq_len,
        pad_mode=pad_mode,
        id_to_label=id_to_label,
        record_fps=args.record_fps,
        min_history=args.min_history,
        smooth=args.smooth,
        silent_when_no_hands=args.silent_when_no_hands,
    )

    tts_worker = TTSWorker()
    tts_worker.start()

    logical_start = time.time()
    prev_loop_time = time.time()
    display_fps = 0.0

    silence_run = 0
    reset_after_silence_frames = 6

    prev_hands = None
    still_run = 0
    stillness_threshold = 0.01
    reset_after_still_frames = 6

    pred_label = ""

    print("Live ASL inference started")
    print("Press q to quit")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Cannot receive frame from webcam")
                break

            now = time.time()
            loop_dt = now - prev_loop_time
            if loop_dt > 0:
                display_fps = 1.0 / loop_dt
            prev_loop_time = now

            timestamp_ms = int((now - logical_start) * 1000)

            result = detector.detect(frame, timestamp_ms=timestamp_ms)
            hands = detector.get_hands_data(result, frame.shape)
            has_hands = len(hands) > 0

            pred_label = predictor.update(hands)

            is_silence_state = (
                (not has_hands)
                or (not pred_label)
                or (pred_label.upper() == "SILENCE")
            )

            if is_silence_state:
                silence_run += 1
            else:
                silence_run = 0

            if silence_run == reset_after_silence_frames:
                predictor.reset()
                tts_worker.reset_speech_state()
                pred_label = ""

            motion = compute_hand_motion(prev_hands, hands)
            prev_hands = hands

            if has_hands and motion < stillness_threshold:
                still_run += 1
            else:
                still_run = 0

            if still_run == reset_after_still_frames:
                predictor.reset()
                tts_worker.reset_speech_state()
                pred_label = ""

            if pred_label and pred_label.upper() != "SILENCE":
                tts_worker.request_speak(pred_label)

            frame_with_landmarks = detector.draw_hands(
                frame.copy(),
                detector.last_result,
            )

            vis_frame = draw_overlay(
                frame=frame_with_landmarks,
                hands_count=len(hands),
                fps=display_fps,
                pred_label=pred_label,
            )

            cv2.imshow("ASL Live Inference", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    finally:
        tts_worker.stop()
        detector.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()