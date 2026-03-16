import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(
        self,
        model_path="models/trained/hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_file}\n"
                f"Hãy tải file hand_landmarker.task và đặt vào đúng đường dẫn này."
            )

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_result = None

    def detect(self, frame, timestamp_ms=None):
        """
        frame: OpenCV BGR image
        timestamp_ms: int, bắt buộc cho VIDEO mode
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_result = result
        return result

    def _get_hand_label_and_score(self, result, hand_idx):
        label = None
        score = None

        if result.handedness and hand_idx < len(result.handedness):
            label = result.handedness[hand_idx][0].category_name
            score = result.handedness[hand_idx][0].score

        return label, score

    def _extract_hand_points(self, hand_landmarks, frame_shape):
        h, w, _ = frame_shape
        points = []

        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        return points

    def draw_hands(self, frame, result):
        """
        Vẽ landmarks và nối các điểm bằng OpenCV thuần.
        """
        if result is None or not result.hand_landmarks:
            return frame

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        hands_data = self.get_hands_data(result, frame.shape)

        for hand_data in hands_data:
            points = hand_data["landmarks"]
            label = hand_data["label"]
            score = hand_data["score"]

            points_tuple = [(p[0], p[1]) for p in points]

            for start_idx, end_idx in connections:
                cv2.line(frame, points_tuple[start_idx], points_tuple[end_idx], (0, 255, 0), 2)

            for x, y in points_tuple:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            text = label or "Unknown"
            if score is not None:
                text += f" {score:.2f}"

            cv2.putText(
                frame,
                text,
                (points_tuple[0][0] + 10, points_tuple[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        return frame

    def get_hands_data(self, result, frame_shape):
        """
        Trả về:
        [
          {
            "label": "Left" | "Right" | None,
            "score": float | None,
            "landmarks": [[x1, y1], ..., [x21, y21]]
          },
          ...
        ]
        """
        hands_data = []

        if result is None or not result.hand_landmarks:
            return hands_data

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            points = self._extract_hand_points(hand_landmarks, frame_shape)
            label, score = self._get_hand_label_and_score(result, i)

            hands_data.append(
                {
                    "label": label,
                    "score": score,
                    "landmarks": points,
                }
            )

        # Giữ thứ tự ổn định giữa các frame
        # Left trước, Right sau, Unknown cuối
        label_order = {"Left": 0, "Right": 1, None: 2}
        hands_data.sort(key=lambda hand: label_order.get(hand["label"], 2))

        return hands_data

    def close(self):
        self.landmarker.close()