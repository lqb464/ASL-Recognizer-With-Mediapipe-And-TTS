from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
import yaml

from src.data.raw_to_interim import process_frame
from src.data.interim_to_processed import FeatureBuilder, sample_to_sequence
from src.models.model import SequenceRNNClassifier


with open("configs/inference.yaml", encoding="utf-8") as f:
    INFER_CFG = yaml.safe_load(f)["infer"]


def load_metadata(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_inference_objects(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> Tuple[SequenceRNNClassifier, FeatureBuilder, int, str, Dict[int, str]]:

    model_path = model_path or Path(INFER_CFG["model_path"])
    meta_path = meta_path or Path(INFER_CFG["meta_path"])

    model = SequenceRNNClassifier.load(model_path)
    meta = load_metadata(meta_path)

    feature_names = meta["feature_names"]
    max_hands = int(meta["max_hands"])
    seq_len = int(meta["seq_len"])
    pad_mode = str(meta["pad_mode"])
    id_to_label = {int(k): str(v) for k, v in meta["id_to_label"].items()}

    feature_builder = FeatureBuilder(
        feature_names=feature_names,
        max_hands=max_hands,
    )

    return model, feature_builder, seq_len, pad_mode, id_to_label


def smooth_label(history: Deque[int], id_to_label: Dict[int, str]) -> str:
    if not history:
        return ""

    counts: Dict[int, int] = {}
    for idx in history:
        counts[idx] = counts.get(idx, 0) + 1

    best_id = max(counts.items(), key=lambda x: x[1])[0]
    return id_to_label.get(best_id, f"id={best_id}")


def build_empty_interim_frame() -> Dict:
    return {"hands": []}


class StreamingPredictor:

    def __init__(
        self,
        model: SequenceRNNClassifier,
        feature_builder: FeatureBuilder,
        seq_len: int,
        pad_mode: str,
        id_to_label: Dict[int, str],
        record_fps: float | None = None,
        min_history: float | None = None,
        smooth: int | None = None,
        silent_when_no_hands: bool | None = None,
    ) -> None:

        self.model = model
        self.feature_builder = feature_builder
        self.seq_len = seq_len
        self.pad_mode = pad_mode
        self.id_to_label = id_to_label

        self.record_fps = float(record_fps or INFER_CFG["record_fps"])
        self.min_history = float(min_history or INFER_CFG["min_history"])
        self.silent_when_no_hands = bool(
            silent_when_no_hands if silent_when_no_hands is not None else INFER_CFG["silent_when_no_hands"]
        )

        smooth = smooth or INFER_CFG["smooth"]

        self.frames_buffer: Deque[Dict] = deque(maxlen=seq_len)
        self.pred_history: Deque[int] = deque(maxlen=max(1, int(smooth)))
        self.min_frames_for_pred = max(1, int(self.min_history * self.record_fps))

    def reset(self) -> None:
        self.frames_buffer.clear()
        self.pred_history.clear()

    def update(self, hands) -> str:

        has_hands = len(hands) > 0

        if has_hands:
            interim_frame = process_frame(hands, score_threshold=0.0)
        else:
            interim_frame = build_empty_interim_frame()

        self.frames_buffer.append(interim_frame)

        if len(self.frames_buffer) < self.min_frames_for_pred:
            return ""

        sample = {
            "sample_id": "live",
            "target": "UNKNOWN",
            "frames": list(self.frames_buffer),
        }

        seq_arr, mask, _ = sample_to_sequence(
            sample=sample,
            feature_builder=self.feature_builder,
            seq_len=self.seq_len,
            pad_mode=self.pad_mode,
        )

        X = seq_arr[np.newaxis, ...]
        m = mask[np.newaxis, ...]

        logits = self.model.forward(X, m)
        pred_id = int(logits.argmax(axis=1)[0])

        self.pred_history.append(pred_id)
        pred_label = smooth_label(self.pred_history, self.id_to_label)

        if self.silent_when_no_hands and not has_hands:
            return ""

        return pred_label