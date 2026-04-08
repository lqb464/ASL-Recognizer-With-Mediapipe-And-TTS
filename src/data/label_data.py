import json
import time
from pathlib import Path

import cv2
import yaml


with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

with open("configs/utils.yaml", encoding="utf-8") as f:
    label_cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
LABEL_CFG = cfg["label"]

RAW_DIR = Path(DATA_CFG["raw_data_dir"])
MANIFEST_PATH = RAW_DIR / DATA_CFG["raw_manifest"]

SILENCE_LABEL = LABEL_CFG["silence_label"]


def init_labeler():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def close_labeler():
    pass


def save_raw_labeled_sample(
    data,
    label: str,
    output_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_id = f"sample_{int(time.time() * 1000)}"
    output_path = output_dir / f"{sample_id}.json"

    sample = {
        "sample_id": sample_id,
        "label": label,
        "num_frames": len(data) if isinstance(data, list) else None,
        "data": data,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)

    manifest_record = {
        "sample_id": sample_id,
        "label": label,
        "raw_file": str(output_path.as_posix()),
        "num_frames": len(data) if isinstance(data, list) else None,
    }

    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    return output_path


def _draw_text(image, text, org, font_scale=0.65, color=(255, 255, 255), thickness=1):
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _get_text_size(text, font_scale=0.65, thickness=1):
    (w, h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    return w, h, baseline


def _fit_text_scale(text, max_width, start_scale=0.65, min_scale=0.42, thickness=1):
    scale = start_scale
    while scale >= min_scale:
        w, _, _ = _get_text_size(text, font_scale=scale, thickness=thickness)
        if w <= max_width:
            return scale
        scale -= 0.02
    return min_scale


def ask_label(preview_frame, num_frames: int | None, window_name="Hand Detection"):
    typed = ""

    while True:
        canvas = preview_frame.copy()
        h, w = canvas.shape[:2]

        blurred = cv2.GaussianBlur(canvas, (15, 15), 0)
        canvas = cv2.addWeighted(canvas, 0.35, blurred, 0.65, 0)

        box_w = int(w * 0.78)
        box_h = int(h * 0.62)

        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (110, 110, 110), 1)

        pad_x = 28
        current_y = y1 + 42
        max_text_width = box_w - pad_x * 2

        _draw_text(
            canvas,
            "ASL LABELING",
            (x1 + pad_x, current_y),
            font_scale=0.9,
            color=(255, 255, 255),
            thickness=2,
        )

        frame_text = f"Frames: {num_frames}"
        frame_scale = 0.62
        frame_w, _, _ = _get_text_size(frame_text, font_scale=frame_scale, thickness=1)

        _draw_text(
            canvas,
            frame_text,
            (x2 - pad_x - frame_w, current_y),
            font_scale=frame_scale,
            color=(210, 210, 210),
            thickness=1,
        )

        divider_y = current_y + 18
        cv2.line(canvas, (x1 + pad_x, divider_y), (x2 - pad_x, divider_y), (90, 90, 90), 1)

        help_lines = [
            "Enter: luu sample (de trong = SILENCE)",
            "skip: bo qua sample nay",
            "ESC: quit    |    Backspace: xoa ky tu",
        ]

        current_y = divider_y + 34
        for line in help_lines:
            scale = _fit_text_scale(line, max_text_width, start_scale=0.62, min_scale=0.42, thickness=1)
            _draw_text(
                canvas,
                line,
                (x1 + pad_x, current_y),
                font_scale=scale,
                color=(235, 235, 235),
                thickness=1,
            )
            _, text_h, _ = _get_text_size(line, font_scale=scale, thickness=1)
            current_y += text_h + 18

        current_y += 8
        _draw_text(
            canvas,
            "Label Input",
            (x1 + pad_x, current_y),
            font_scale=0.62,
            color=(220, 220, 220),
            thickness=1,
        )

        input_box_y1 = current_y + 16
        input_box_y2 = input_box_y1 + 48
        input_box_x1 = x1 + pad_x
        input_box_x2 = x2 - pad_x

        cv2.rectangle(canvas, (input_box_x1, input_box_y1), (input_box_x2, input_box_y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (input_box_x1, input_box_y1), (input_box_x2, input_box_y2), (120, 120, 120), 1)

        blink = (int(time.time() * 2) % 2) == 0
        display_text = typed + ("|" if blink else "")

        text_scale = _fit_text_scale(
            display_text if display_text else " ",
            input_box_x2 - input_box_x1 - 20,
            start_scale=0.7,
            min_scale=0.45,
            thickness=1,
        )

        _draw_text(
            canvas,
            display_text,
            (input_box_x1 + 10, input_box_y1 + 31),
            font_scale=text_scale,
            color=(255, 255, 255),
            thickness=1,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == 13:
            value = typed.strip()
            if not value:
                return SILENCE_LABEL
            return value

        if key == 27:
            return 27

        if key in (8, 127):
            typed = typed[:-1]
            continue

        if key == 32:
            typed += " "
            continue

        if 32 <= key <= 126:
            typed += chr(key)