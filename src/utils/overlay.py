from __future__ import annotations

import cv2


def draw_overlay(
    frame,
    hands_count: int,
    fps: float,
    pred_label: str,
):
    """
    Draw a minimal live inference overlay.
    """
    vis_frame = frame.copy()

    cv2.putText(
        vis_frame,
        f"Hands: {hands_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis_frame,
        f"FPS: {fps:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    if pred_label and pred_label.upper() != "SILENCE":
        cv2.putText(
            vis_frame,
            f"PRED: {pred_label}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis_frame