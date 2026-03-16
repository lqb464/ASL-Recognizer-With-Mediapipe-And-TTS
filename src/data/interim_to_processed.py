from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
PROCESSING_CFG = cfg["processing"]

DEFAULT_INPUT = Path(DATA_CFG["interim_data_dir"])
DEFAULT_OUTPUT = Path(DATA_CFG["processed_data_dir"]) / PROCESSING_CFG["default_output_name"]
DEFAULT_FEATURES = list(PROCESSING_CFG["default_features"])

FEATURE_SIZES = {
    "coords_norm_flat": 42,
    "bone_vectors_flat": 40,
    "joint_angles": 10,
    "tip_distances": 7,
    "handedness": 1,
    "handedness_onehot": 3,
    "present": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all interim JSON samples into one processed train file for sequence models."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Interim JSON directory or a single JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Single output dataset file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--format",
        default=PROCESSING_CFG["default_output_format"],
        choices=["npz", "pt"],
        help=f"Output format. Default: {PROCESSING_CFG['default_output_format']}",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=int(PROCESSING_CFG["default_seq_len"]),
        help=(
            "Fixed sequence length. Use 0 to auto-detect from the longest sample in input. "
            f"Default: {PROCESSING_CFG['default_seq_len']}"
        ),
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=int(PROCESSING_CFG["default_max_hands"]),
        choices=[1, 2],
        help=f"Maximum hands per frame to keep. Default: {PROCESSING_CFG['default_max_hands']}",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help=(
            "Per-hand features to use. Available: coords_norm_flat bone_vectors_flat "
            "joint_angles tip_distances handedness handedness_onehot present"
        ),
    )
    parser.add_argument(
        "--pad-mode",
        default=PROCESSING_CFG["default_pad_mode"],
        choices=["zero", "repeat_last", "loop"],
        help=f"Padding mode for shorter sequences. Default: {PROCESSING_CFG['default_pad_mode']}",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path. Default: same folder as output, with _meta suffix",
    )
    return parser.parse_args()


def discover_json_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.json"))


def load_interim_samples(files: Sequence[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        if not isinstance(sample, dict):
            raise ValueError(f"Invalid JSON root in file: {file_path}")
        samples.append(sample)
    return samples


def build_label_maps(samples: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({str(sample.get("target", "UNKNOWN")) for sample in samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def handedness_onehot(handedness: int) -> List[float]:
    if handedness == 0:
        return [1.0, 0.0, 0.0]
    if handedness == 1:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class FeatureBuilder:
    def __init__(self, feature_names: Sequence[str], max_hands: int) -> None:
        self.feature_names = list(feature_names)
        self.max_hands = max_hands
        unknown = [name for name in self.feature_names if name not in FEATURE_SIZES]
        if unknown:
            raise ValueError(f"Unknown feature names: {unknown}")
        self.per_hand_dim = sum(FEATURE_SIZES[name] for name in self.feature_names)
        self.frame_dim = self.per_hand_dim * self.max_hands

    def _read_vector_feature(self, hand: Dict[str, Any], name: str) -> List[float]:
        values = hand.get(name, [])
        if not isinstance(values, list):
            raise ValueError(f"Feature '{name}' is missing or is not a list")
        expected_dim = FEATURE_SIZES[name]
        if len(values) != expected_dim:
            raise ValueError(
                f"Feature '{name}' has wrong size. Expected {expected_dim}, got {len(values)}"
            )
        return [float(v) for v in values]

    def hand_to_vector(self, hand: Dict[str, Any]) -> np.ndarray:
        out: List[float] = []
        for name in self.feature_names:
            if name == "handedness_onehot":
                out.extend(handedness_onehot(int(hand.get("handedness", -1))))
            elif name == "present":
                out.append(float(hand.get("present", 0)))
            elif name == "handedness":
                out.append(float(hand.get("handedness", -1)))
            else:
                out.extend(self._read_vector_feature(hand, name))
        return np.asarray(out, dtype=np.float32)

    def empty_hand_vector(self) -> np.ndarray:
        return np.zeros(self.per_hand_dim, dtype=np.float32)

    def frame_to_vector(self, frame: Dict[str, Any]) -> np.ndarray:
        hands = frame.get("hands", [])
        if not isinstance(hands, list):
            hands = []

        vectors: List[np.ndarray] = []
        for hand in hands[: self.max_hands]:
            vectors.append(self.hand_to_vector(hand))

        while len(vectors) < self.max_hands:
            vectors.append(self.empty_hand_vector())

        return np.concatenate(vectors, axis=0)


def infer_seq_len(samples: Sequence[Dict[str, Any]], user_seq_len: int) -> int:
    if user_seq_len and user_seq_len > 0:
        return user_seq_len
    longest = max((len(sample.get("frames", [])) for sample in samples), default=1)
    return max(longest, 1)


def sample_to_sequence(
    sample: Dict[str, Any],
    feature_builder: FeatureBuilder,
    seq_len: int,
    pad_mode: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    frames = sample.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    frame_vectors = [feature_builder.frame_to_vector(frame) for frame in frames]
    original_len = len(frame_vectors)

    if original_len == 0:
        sequence = np.zeros((seq_len, feature_builder.frame_dim), dtype=np.float32)
        mask = np.zeros(seq_len, dtype=np.float32)
        return sequence, mask, 0

    if original_len >= seq_len:
        sequence = np.stack(frame_vectors[:seq_len], axis=0).astype(np.float32)
        mask = np.ones(seq_len, dtype=np.float32)
        return sequence, mask, seq_len

    sequence = np.zeros((seq_len, feature_builder.frame_dim), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)

    stacked = np.stack(frame_vectors, axis=0).astype(np.float32)
    sequence[:original_len] = stacked
    mask[:original_len] = 1.0

    if pad_mode == "repeat_last":
        sequence[original_len:] = stacked[-1]
    elif pad_mode == "loop":
        for idx in range(original_len, seq_len):
            sequence[idx] = stacked[idx % original_len]

    return sequence, mask, original_len


def build_dataset(
    samples: Sequence[Dict[str, Any]],
    feature_builder: FeatureBuilder,
    seq_len: int,
    pad_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    label_to_id, id_to_label = build_label_maps(samples)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    lengths_list: List[int] = []
    masks_list: List[np.ndarray] = []
    sample_ids: List[str] = []

    for sample in samples:
        sequence, mask, effective_len = sample_to_sequence(
            sample=sample,
            feature_builder=feature_builder,
            seq_len=seq_len,
            pad_mode=pad_mode,
        )
        label = str(sample.get("target", "UNKNOWN"))
        X_list.append(sequence)
        y_list.append(label_to_id[label])
        lengths_list.append(effective_len)
        masks_list.append(mask)
        sample_ids.append(str(sample.get("sample_id", "unknown_sample")))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    lengths = np.asarray(lengths_list, dtype=np.int64)
    masks = np.stack(masks_list, axis=0).astype(np.float32)
    sample_ids_arr = np.asarray(sample_ids)
    return X, y, lengths, masks, sample_ids_arr, label_to_id, id_to_label


def save_npz(
    output_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    masks: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        lengths=lengths,
        masks=masks,
        sample_ids=sample_ids,
    )


def save_pt(output_path: Path, payload: Dict[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Use --format npz or install torch first."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def write_metadata(
    meta_path: Path,
    input_path: Path,
    files: Sequence[Path],
    seq_len: int,
    pad_mode: str,
    feature_builder: FeatureBuilder,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    num_samples: int,
    output_format: str,
) -> None:
    metadata = {
        "input": str(input_path),
        "num_input_files": len(files),
        "input_files": [str(p) for p in files],
        "num_samples": num_samples,
        "seq_len": seq_len,
        "max_hands": feature_builder.max_hands,
        "pad_mode": pad_mode,
        "feature_names": feature_builder.feature_names,
        "per_hand_dim": feature_builder.per_hand_dim,
        "frame_dim": feature_builder.frame_dim,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "output_format": output_format,
        "description": {
            "X": "shape=(N, T, F), sequence tensor ready for RNN/LSTM/GRU/Transformer",
            "y": "shape=(N,), integer class labels",
            "lengths": "shape=(N,), valid frame count before padding/truncation",
            "masks": "shape=(N, T), 1=valid frame, 0=padded frame",
            "sample_ids": "shape=(N,), sample identifiers",
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    files = discover_json_files(input_path)
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {input_path}")

    samples = load_interim_samples(files)
    feature_builder = FeatureBuilder(args.features, max_hands=args.max_hands)
    seq_len = infer_seq_len(samples, args.seq_len)

    X, y, lengths, masks, sample_ids, label_to_id, id_to_label = build_dataset(
        samples=samples,
        feature_builder=feature_builder,
        seq_len=seq_len,
        pad_mode=args.pad_mode,
    )

    if args.format == "npz":
        save_npz(output_path, X, y, lengths, masks, sample_ids)
    else:
        payload = {
            "X": X,
            "y": y,
            "lengths": lengths,
            "masks": masks,
            "sample_ids": sample_ids,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "feature_names": feature_builder.feature_names,
            "frame_dim": feature_builder.frame_dim,
            "seq_len": seq_len,
        }
        save_pt(output_path, payload)

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else output_path.with_name(output_path.stem + "_meta.json")
    )
    write_metadata(
        meta_path=metadata_path,
        input_path=input_path,
        files=files,
        seq_len=seq_len,
        pad_mode=args.pad_mode,
        feature_builder=feature_builder,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        num_samples=len(samples),
        output_format=args.format,
    )

    print(f"Loaded {len(files)} interim files")
    print(f"Built single processed dataset: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"masks shape: {masks.shape}")
    print(f"frame_dim: {feature_builder.frame_dim}")
    print(f"seq_len: {seq_len}")
    print(f"label_to_id: {label_to_id}")


if __name__ == "__main__":
    main()