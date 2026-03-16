from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import cv2

from src.data.label_data import RAW_DIR, MANIFEST_PATH, save_raw_labeled_sample

if TYPE_CHECKING:
    from src.utils.hand_detector import HandDetector


import yaml

with open("configs/data.yaml") as f:
    cfg = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Import external videos -> run MediaPipe -> save raw samples into data/raw as JSON.\n"
            "This produces the same raw format as webcam collection, so you can reuse raw_to_interim.py."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(cfg["data"]["external_data_dir"]),
        help="Video file or directory containing videos. Default: data/external",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.mp4",
        help="When --input is a directory, match videos using this glob. Default: *.mp4",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label to apply to all imported videos. If omitted, use --label-from-parent or --labels-json.",
    )
    p.add_argument(
        "--label-from-parent",
        action="store_true",
        help="Infer label from the immediate parent folder name of each video.",
    )
    p.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping for labels. Keys can be filename (e.g. clip1.mp4) or "
            "relative path from --input. Values are label strings."
        ),
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=cfg["record"]["record_fps"],
        help=f"Sample frames at this FPS from the video. Default: {cfg['record']['record_fps']}",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Optional cap per video in seconds (0 = no cap).",
    )
    p.add_argument(
        "--num-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Maximum hands for MediaPipe to detect. Default: 2",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="models/trained/hand_landmarker.task",
        help="Path to MediaPipe hand landmarker .task file.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be imported; do not write data/raw outputs.",
    )
    return p.parse_args()


def discover_videos(input_path: Path, glob_pat: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []
    return sorted([p for p in input_path.rglob(glob_pat) if p.is_file()])


def load_labels_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"labels-json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("--labels-json must be a JSON object mapping path->label")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        out[str(k)] = str(v)
    return out


def resolve_label(
    video_path: Path,
    input_root: Path,
    fixed_label: Optional[str],
    label_from_parent: bool,
    labels_map: Dict[str, str],
) -> str:
    if fixed_label:
        return fixed_label

    rel = None
    if input_root.exists() and input_root.is_dir():
        try:
            rel = str(video_path.relative_to(input_root).as_posix())
        except ValueError:
            rel = None

    key_candidates = []
    if rel:
        key_candidates.append(rel)
    key_candidates.append(video_path.name)

    for key in key_candidates:
        if key in labels_map:
            return str(labels_map[key])

    if label_from_parent:
        return video_path.parent.name

    raise ValueError(
        "Cannot resolve label. Provide --label, or --label-from-parent, or --labels-json."
    )


def iter_sampled_frames(
    cap: cv2.VideoCapture, record_fps: float, max_seconds: float
) -> Iterable[Tuple[int, float, "cv2.MatLike"]]:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-6:
        src_fps = 30.0

    stride = max(int(round(src_fps / max(record_fps, 1e-6))), 1)
    max_frames = 0
    if max_seconds and max_seconds > 0:
        max_frames = int(max_seconds * record_fps)

    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % stride == 0:
            ts_s = kept / max(record_fps, 1e-6)
            yield kept, ts_s, frame
            kept += 1
            if max_frames and kept >= max_frames:
                break

        frame_idx += 1


def process_video(
    video_path: Path,
    label: str,
    detector: HandDetector,
    record_fps: float,
    max_seconds: float,
) -> List[List[Dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        sequence: List[List[Dict]] = []
        # Use VIDEO mode timestamps; keep them strictly increasing.
        base_ms = int(time.time() * 1000)
        for kept_idx, ts_s, frame in iter_sampled_frames(cap, record_fps, max_seconds):
            timestamp_ms = base_ms + int(ts_s * 1000.0)
            result = detector.detect(frame, timestamp_ms=timestamp_ms)
            hands = detector.get_hands_data(result, frame.shape)
            sequence.append(hands)

        if len(sequence) == 0:
            raise RuntimeError(f"No frames sampled from: {video_path}")
        return sequence
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.input, args.glob)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {args.input} (glob={args.glob})")

    labels_map = load_labels_map(args.labels_json)
    from src.utils.hand_detector import HandDetector

    detector = HandDetector(model_path=args.model_path, num_hands=args.num_hands)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0

    try:
        for vp in videos:
            try:
                label = resolve_label(
                    video_path=vp,
                    input_root=args.input,
                    fixed_label=args.label,
                    label_from_parent=args.label_from_parent,
                    labels_map=labels_map,
                )

                print(f"[IMPORT] {vp} -> label='{label}'")
                if args.dry_run:
                    skipped += 1
                    continue

                seq = process_video(
                    video_path=vp,
                    label=label,
                    detector=detector,
                    record_fps=float(args.record_fps),
                    max_seconds=float(args.max_seconds),
                )
                out_path = save_raw_labeled_sample(data=seq, label=label)
                imported += 1
                print(f"  saved: {out_path} (frames={len(seq)})")
            except Exception as exc:
                skipped += 1
                print(f"[SKIP] {vp}: {exc}")

    finally:
        detector.close()

    print(f"Done. imported={imported} skipped={skipped} raw_dir={RAW_DIR}")


if __name__ == "__main__":
    main()

