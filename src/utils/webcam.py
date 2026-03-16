import cv2


class Webcam:
    def __init__(self, camera_index=0, width=None, height=None, fps=None):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        return frame

    def get_actual_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            return None
        return fps

    def release(self):
        self.cap.release()