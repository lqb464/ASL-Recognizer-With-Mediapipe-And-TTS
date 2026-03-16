from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
LANDMARKS_CFG = cfg["landmarks"]
LABEL_CFG = cfg["label"]

DEFAULT_INPUT = Path(DATA_CFG["raw_data_dir"])
DEFAULT_OUTPUT = Path(DATA_CFG["interim_data_dir"])

WRIST = int(LANDMARKS_CFG["wrist"])
INDEX_MCP = int(LANDMARKS_CFG["index_mcp"])
MIDDLE_MCP = int(LANDMARKS_CFG["middle_mcp"])
RING_MCP = int(LANDMARKS_CFG["ring_mcp"])
PINKY_MCP = int(LANDMARKS_CFG["pinky_mcp"])

THUMB_TIP = int(LANDMARKS_CFG["thumb_tip"])
INDEX_TIP = int(LANDMARKS_CFG["index_tip"])
MIDDLE_TIP = int(LANDMARKS_CFG["middle_tip"])
RING_TIP = int(LANDMARKS_CFG["ring_tip"])
PINKY_TIP = int(LANDMARKS_CFG["pinky_tip"])

NUM_LANDMARKS = int(LANDMARKS_CFG["num_landmarks"])
EPS = float(LABEL_CFG["eps"])

BONES: List[Tuple[int, int]] = [tuple(pair) for pair in LANDMARKS_CFG["bones"]]
ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    tuple(triplet) for triplet in LANDMARKS_CFG["angle_triplets"]
]
TIP_PAIRS: List[Tuple[int, int]] = [tuple(pair) for pair in LANDMARKS_CFG["tip_pairs"]]


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def _safe_scale(points: List[Tuple[float, float]]) -> float:
    wrist = points[WRIST]
    middle_mcp = points[MIDDLE_MCP]
    scale = _distance(wrist, middle_mcp)
    if scale > EPS:
        return scale

    palm_candidates = [
        _distance(points[INDEX_MCP], points[PINKY_MCP]),
        _distance(points[INDEX_MCP], points[RING_MCP]),
        _distance(points[MIDDLE_MCP], points[PINKY_MCP]),
    ]
    scale = max(palm_candidates)
    if scale > EPS:
        return scale

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox_diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    return bbox_diag if bbox_diag > EPS else 1.0


def normalize_landmarks(points: List[List[float]]) -> List[List[float]]:
    pts = [(float(x), float(y)) for x, y in points]
    origin = pts[WRIST]
    scale = _safe_scale(pts)
    return [[(x - origin[0]) / scale, (y - origin[1]) / scale] for x, y in pts]


def bone_vectors(norm_points: List[List[float]]) -> List[List[float]]:
    return [
        [
            norm_points[j][0] - norm_points[i][0],
            norm_points[j][1] - norm_points[i][1],
        ]
        for i, j in BONES
    ]


def joint_angles(norm_points: List[List[float]]) -> List[float]:
    out: List[float] = []

    for a, b, c in ANGLE_TRIPLETS:
        bax = norm_points[a][0] - norm_points[b][0]
        bay = norm_points[a][1] - norm_points[b][1]
        bcx = norm_points[c][0] - norm_points[b][0]
        bcy = norm_points[c][1] - norm_points[b][1]

        norm_ba = (bax * bax + bay * bay) ** 0.5
        norm_bc = (bcx * bcx + bcy * bcy) ** 0.5
        if norm_ba < EPS or norm_bc < EPS:
            out.append(0.0)
            continue

        cos_theta = (bax * bcx + bay * bcy) / (norm_ba * norm_bc)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        out.append(math.acos(cos_theta))

    return out


def tip_distances(norm_points: List[List[float]]) -> List[float]:
    distances: List[float] = []
    for i, j in TIP_PAIRS:
        distances.append(_distance(tuple(norm_points[i]), tuple(norm_points[j])))
    return distances


def handedness_to_id(name: Optional[str]) -> int:
    if not name:
        return -1

    lowered = name.strip().lower()
    if lowered == "left":
        return 0
    if lowered == "right":
        return 1
    return -1


def flatten_2d(points: List[List[float]]) -> List[float]:
    return [coord for point in points for coord in point]


def process_hand(hand: Dict[str, Any], score_threshold: float) -> Optional[Dict[str, Any]]:
    score = float(hand.get("score", 1.0))
    landmarks = hand.get("landmarks")

    if score < score_threshold:
        return None
    if not isinstance(landmarks, list) or len(landmarks) != NUM_LANDMARKS:
        return None
    if any(not isinstance(p, list) or len(p) != 2 for p in landmarks):
        return None

    coords_norm = normalize_landmarks(landmarks)
    vectors = bone_vectors(coords_norm)
    angles = joint_angles(coords_norm)
    dists = tip_distances(coords_norm)

    return {
        "handedness": handedness_to_id(hand.get("label")),
        "handedness_name": hand.get("label", "Unknown"),
        "present": 1,
        "coords_norm": coords_norm,
        "coords_norm_flat": flatten_2d(coords_norm),
        "bone_vectors": vectors,
        "bone_vectors_flat": flatten_2d(vectors),
        "joint_angles": angles,
        "tip_distances": dists,
    }


def process_frame(frame_hands: List[Dict[str, Any]], score_threshold: float) -> Dict[str, Any]:
    processed_hands: List[Dict[str, Any]] = []

    for hand in frame_hands:
        processed = process_hand(hand, score_threshold=score_threshold)
        if processed is not None:
            processed_hands.append(processed)

    processed_hands.sort(key=lambda h: (h["handedness"] == -1, h["handedness"]))

    return {
        "num_hands": len(processed_hands),
        "hands": processed_hands,
    }


def convert_sample(raw_sample: Dict[str, Any], score_threshold: float) -> Dict[str, Any]:
    frames = raw_sample.get("data", [])
    interim_frames: List[Dict[str, Any]] = []

    for frame_idx, frame_hands in enumerate(frames):
        if not isinstance(frame_hands, list):
            frame_hands = []

        frame_data = process_frame(frame_hands, score_threshold=score_threshold)
        frame_data["frame_idx"] = frame_idx
        interim_frames.append(frame_data)

    return {
        "sample_id": raw_sample.get("sample_id"),
        "target": raw_sample.get("label"),
        "num_frames": len(interim_frames),
        "feature_schema": {
            "coords_norm": "21 normalized (x, y) landmarks using wrist as origin and wrist->middle_mcp as scale",
            "bone_vectors": "20 normalized 2D bone vectors",
            "joint_angles": "10 joint angles in radians",
            "tip_distances": "7 normalized fingertip distances",
            "handedness": "0=Left, 1=Right, -1=Unknown",
        },
        "frames": interim_frames,
    }


def convert_file(input_path: Path, output_path: Path, score_threshold: float) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        raw_sample = json.load(f)

    interim_sample = convert_sample(raw_sample, score_threshold=score_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(interim_sample, f, ensure_ascii=False, indent=2)


def discover_json_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw hand-landmark ASL samples into interim normalized features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON file or folder. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file or folder. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=float(LABEL_CFG["threshold_score"]),
        help=f"Drop hands whose detector score is below this value. Default: {LABEL_CFG['threshold_score']}",
    )
    args = parser.parse_args()

    input_files = discover_json_files(args.input)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in: {args.input}")

    single_file_mode = args.input.is_file()

    for input_file in input_files:
        if single_file_mode:
            output_file = args.output
            if output_file.suffix.lower() != ".json":
                output_file = output_file / input_file.name
        else:
            output_file = args.output / input_file.name

        convert_file(
            input_path=input_file,
            output_path=output_file,
            score_threshold=args.score_threshold,
        )
        print(f"Converted: {input_file} -> {output_file}")


if __name__ == "__main__":
    main()