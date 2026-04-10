import argparse
import json
from pathlib import Path
import numpy as np
import yaml

with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
PROC_CFG = cfg["processing"]
DEFAULT_INPUT = Path(DATA_CFG["interim_data_dir"])
DEFAULT_OUTPUT = Path(DATA_CFG["processed_data_dir"]) / PROC_CFG["default_output_name"]

def extract_features(sample_data: list):
    """Biến đổi cấu trúc list/dict thành mảng numpy phẳng."""
    sequence_features = []
    for frame in sample_data:
        # Mặc định lấy tay đầu tiên nếu có, nếu không thì điền 0
        # (Bạn có thể mở rộng logic này tùy theo yêu cầu của model)
        if len(frame) > 0:
            hand = frame[0]
            lmks = hand["landmarks"]
            # Flatten 21 landmarks x 3 (x,y,z) = 63 features
            flat_lmks = []
            for p in lmks:
                flat_lmks.extend([p["x"], p["y"], p["z"]])
        else:
            flat_lmks = [0.0] * 63
        sequence_features.append(flat_lmks)
    return np.array(sequence_features, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_samples = []
    for f_path in args.input.glob("*.jsonl"):
        with f_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

    if not all_samples:
        print("Không có dữ liệu để xử lý.")
        return

    X, y, ids = [], [], []
    labels_map = {}
    
    # Tạo label mapping tự động
    unique_labels = sorted(list(set(s["label"] for s in all_samples)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    for s in all_samples:
        feat = extract_features(s["data"])
        X.append(feat)
        y.append(label_to_id[s["label"]])
        ids.append(s["sample_id"])

    # Lưu file nén NPZ
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.array(X),
        y=np.array(y),
        sample_ids=np.array(ids),
        label_map=json.dumps(label_to_id)
    )
    print(f"Đã lưu dataset cuối cùng ({len(X)} mẫu) tại: {args.output}")