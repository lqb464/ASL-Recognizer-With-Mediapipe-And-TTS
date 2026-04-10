import argparse
import json
import math
from pathlib import Path
import yaml

# Tải cấu hình
with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
LANDMARKS_CFG = cfg["landmarks"]
LABEL_CFG = cfg["label"]

DEFAULT_INPUT = Path(DATA_CFG["raw_data_dir"])
DEFAULT_OUTPUT = Path(DATA_CFG["interim_data_dir"])

# Các hằng số landmark từ config
WRIST = int(LANDMARKS_CFG["wrist"])
NUM_LANDMARKS = int(LANDMARKS_CFG["num_landmarks"])
EPS = float(LABEL_CFG["eps"])


def normalize_landmarks(hand_data: dict):
    """Chuẩn hóa tọa độ một bàn tay về gốc tọa độ Wrist và scale theo kích thước bàn tay."""
    if not hand_data or "landmarks" not in hand_data:
        return None

    lmks = hand_data["landmarks"]
    if not lmks or len(lmks) != NUM_LANDMARKS:
        return None

    wrist_pt = lmks[WRIST]

    # 1. Tịnh tiến về Wrist (0,0,0)
    translated = []
    for p in lmks:
        translated.append({
            "x": p["x"] - wrist_pt["x"],
            "y": p["y"] - wrist_pt["y"],
            "z": p["z"] - wrist_pt["z"]
        })

    # 2. Tìm khoảng cách xa nhất từ Wrist để scale
    max_dist = 0.0
    for p in translated:
        dist = math.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
        if dist > max_dist:
            max_dist = dist

    scale = 1.0 / (max_dist + EPS)

    normalized = []
    for p in translated:
        normalized.append({
            "x": p["x"] * scale,
            "y": p["y"] * scale,
            "z": p["z"] * scale
        })
    return normalized


def process_sample(raw_sample: dict, score_threshold: float):
    """Xử lý toàn bộ các frame trong một clip."""
    processed_frames = []

    for frame_hands in raw_sample["data"]:
        frame_result = []

        for hand in frame_hands:
            if hand["score"] < score_threshold:
                continue

            norm_lmks = normalize_landmarks(hand)
            if norm_lmks:
                frame_result.append({
                    "handedness": hand["label"],
                    "score": hand["score"],
                    "landmarks": norm_lmks
                })

        # Chỉ giữ frame nếu frame đó còn hand hợp lệ
        if len(frame_result) > 0:
            processed_frames.append(frame_result)

    # Nếu toàn bộ frame đều rỗng sau xử lý thì loại sample
    if len(processed_frames) == 0:
        return None

    return {
        "sample_id": raw_sample["sample_id"],
        "label": raw_sample["label"],
        "data": processed_frames
    }


def convert_file(input_path: Path, output_path: Path, score_threshold: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    dropped_count = 0

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.with_suffix(".jsonl").open("w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            raw_sample = json.loads(line)
            interim_sample = process_sample(raw_sample, score_threshold)

            if interim_sample is None:
                dropped_count += 1
                continue

            f_out.write(json.dumps(interim_sample, ensure_ascii=False) + "\n")
            kept_count += 1

    print(
        f"-> Đã chuyển đổi: {input_path.name} | "
        f"Giữ: {kept_count} mẫu | Loại: {dropped_count} mẫu rỗng"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-threshold", type=float, default=float(LABEL_CFG["threshold_score"]))
    args = parser.parse_args()

    input_files = list(args.input.glob("*.jsonl"))
    if not input_files:
        print(f"Không tìm thấy file .jsonl nào tại {args.input}")
        return

    for in_file in input_files:
        out_file = args.output / in_file.name
        convert_file(in_file, out_file, args.score_threshold)