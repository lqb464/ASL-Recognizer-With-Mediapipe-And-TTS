import math
import time

import cv2
import yaml

from ..data.label_data import (
    ask_label,
    close_labeler,
    init_labeler,
    save_raw_labeled_sample,
)
from ..utils.hand_detector import HandDetector
from ..utils.webcam import Webcam


with open("configs/data.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

with open("configs/utils.yaml", encoding="utf-8") as f:
    utils_cfg = yaml.safe_load(f)

CAMERA_CFG = utils_cfg["webcam"]
RECORD_CFG = data_cfg["record"]

CAMERA_INDEX = int(CAMERA_CFG["index"])
CAMERA_WIDTH = int(CAMERA_CFG["width"])
CAMERA_HEIGHT = int(CAMERA_CFG["height"])
CAMERA_FPS = int(CAMERA_CFG["fps"])

COUNTDOWN_SECONDS = float(RECORD_CFG["countdown_seconds"])
RECORD_SECONDS = float(RECORD_CFG["record_seconds"])
RECORD_FPS = float(RECORD_CFG["record_fps"])

WINDOW_NAME = "Hand Detection"


def main():
    cam = Webcam(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )
    detector = HandDetector()
    init_labeler()

    actual_camera_fps = cam.get_actual_fps()

    print("Press 'q' to quit")
    print(f"Auto countdown: {COUNTDOWN_SECONDS}s")
    print(f"Auto record duration: {RECORD_SECONDS}s")
    print(f"Requested camera config: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
    print(f"Actual camera FPS reported by OpenCV: {actual_camera_fps}")
    print(f"Record FPS: {RECORD_FPS}")

    start_time = time.time()
    prev_hands = None

    sequence = []
    countdown_active = True
    recording = False

    countdown_start_time = time.time()
    record_start_time = None
    last_record_time = 0.0

    should_quit = False
    last_saved_label = None

    prev_loop_time = time.time()
    display_fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Cannot receive frame")
                break

            now = time.time()
            loop_dt = now - prev_loop_time
            if loop_dt > 0:
                display_fps = 1.0 / loop_dt
            prev_loop_time = now

            timestamp_ms = int((time.time() - start_time) * 1000)

            result = detector.detect(frame, timestamp_ms=timestamp_ms)
            frame = detector.draw_hands(frame, result)

            hands = detector.get_hands_data(result, frame.shape)
            current_hands = len(hands)

            if current_hands != prev_hands:
                if current_hands == 0:
                    print("No hands detected")
                else:
                    print(f"Detected hands: {current_hands}")
                prev_hands = current_hands

            status = "IDLE"

            if countdown_active and not recording:
                elapsed = time.time() - countdown_start_time
                remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
                countdown_number = max(0, math.ceil(remaining))
                status = f"COUNTDOWN {countdown_number}"

                if elapsed >= COUNTDOWN_SECONDS:
                    countdown_active = False
                    recording = True
                    sequence = []
                    record_start_time = time.time()
                    last_record_time = 0.0
                    status = "RECORDING"
                    print("Start recording...")

            elif recording:
                current_time = time.time()
                status = f"RECORDING | frames={len(sequence)}"

                if current_time - last_record_time >= 1.0 / RECORD_FPS:
                    sequence.append(hands)
                    last_record_time = current_time

                if record_start_time is not None and (current_time - record_start_time) >= RECORD_SECONDS:
                    recording = False
                    print(f"Stop recording. Captured {len(sequence)} frames.")

                    preview_frame = frame.copy()
                    user_input = ask_label(
                        preview_frame=preview_frame,
                        num_frames=len(sequence),
                        window_name=WINDOW_NAME,
                    )

                    if user_input == 27:
                        print("Quit by label dialog.")
                        should_quit = True

                    elif user_input.lower() == "skip":
                        print("Skipped sample.")

                    else:
                        output_path = save_raw_labeled_sample(
                            data=sequence,
                            label=user_input,
                        )
                        last_saved_label = user_input
                        print(f"Saved sample to: {output_path}")

                    if should_quit:
                        break

                    countdown_active = True
                    countdown_start_time = time.time()
                    record_start_time = None
                    sequence = []
                    last_record_time = 0.0
                    print("Next countdown started...")

            else:
                status = "IDLE"

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Hands: {current_hands}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if last_saved_label is not None:
                cv2.putText(
                    frame,
                    f"Last label: {last_saved_label}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            if countdown_active and not recording:
                helper = "Get ready... recording starts automatically"
            elif recording:
                helper = "Recording..."
            else:
                helper = "Waiting..."

            cv2.putText(
                frame,
                helper,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Quit by keyboard.")
                break

    finally:
        close_labeler()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()