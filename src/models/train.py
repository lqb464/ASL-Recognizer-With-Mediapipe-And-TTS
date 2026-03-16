from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.model import SequenceRNNClassifier, SequenceRNNConfig, accuracy, save_json


DEFAULT_DATASET = Path("data/processed/train.npz")
DEFAULT_OUT_DIR = Path("models/checkpoints")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a temporal RNN classifier on processed ASL dataset.")
    p.add_argument("--data", default=str(DEFAULT_DATASET), help="Path to processed dataset .npz")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for checkpoints")
    p.add_argument("--model-type", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    X = obj["X"].astype(np.float32)
    y = obj["y"].astype(np.int64)
    lengths = obj["lengths"].astype(np.int64) if "lengths" in obj else np.zeros((len(y),), np.int64)
    masks = obj["masks"].astype(np.float32)
    sample_ids = obj["sample_ids"] if "sample_ids" in obj else np.asarray([f"s{i}" for i in range(len(y))])
    return X, y, lengths, masks, sample_ids


def split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(round(n * val_split))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx


def iterate_minibatches(
    X: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    order = np.arange(n)
    rng.shuffle(order)
    for start in range(0, n, batch_size):
        sl = order[start : start + batch_size]
        yield X[sl], masks[sl], y[sl]


def evaluate(
    model: SequenceRNNClassifier,
    X: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    preds_all = []

    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = X[start : start + batch_size]
            mb = masks[start : start + batch_size]

            xb_t = torch.from_numpy(xb).to(device)
            mb_t = torch.from_numpy(mb).to(device)

            logits = model.forward_torch(xb_t, mb_t)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            preds_all.append(preds)

    y_pred = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0,), dtype=np.int64)
    acc = accuracy(y, y_pred)
    return {"acc": acc}


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Build it via src/data/interim_to_processed.py"
        )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, lengths, masks, sample_ids = load_npz_dataset(data_path)
    n, t, f = X.shape
    num_classes = int(y.max() + 1) if y.size else 0

    if num_classes <= 1:
        raise ValueError(f"Need at least 2 classes to train, got {num_classes}")

    train_idx, val_idx = split_indices(n, args.val_split, args.seed)
    X_tr, m_tr, y_tr = X[train_idx], masks[train_idx], y[train_idx]
    X_va, m_va, y_va = X[val_idx], masks[val_idx], y[val_idx]

    model = SequenceRNNClassifier(
        SequenceRNNConfig(
            input_dim=f,
            num_classes=num_classes,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            seed=args.seed,
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = out_dir / f"asl_{args.model_type}_best.pt"
    meta_path = out_dir / f"asl_{args.model_type}_best.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        losses = []
        accs = []

        for xb, mb, yb in iterate_minibatches(
            X_tr,
            m_tr,
            y_tr,
            args.batch_size,
            seed=args.seed + epoch,
        ):
            xb_t = torch.from_numpy(xb).to(device)
            mb_t = torch.from_numpy(mb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            optimizer.zero_grad()
            logits = model.forward_torch(xb_t, mb_t)
            loss = criterion(logits, yb_t)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            acc = accuracy(yb, preds)

            losses.append(float(loss.item()))
            accs.append(acc)

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = float(np.mean(accs)) if accs else 0.0
        val_metrics = evaluate(model, X_va, m_va, y_va, device=device)
        dt = time.time() - t0

        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['acc']:.4f} time={dt:.2f}s"
        )

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            extra = {
                "dataset": str(data_path.as_posix()),
                "shapes": {"X": [int(n), int(t), int(f)]},
                "val_split": float(args.val_split),
                "seed": int(args.seed),
                "model_type": str(args.model_type),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "bidirectional": bool(args.bidirectional),
            }
            model.save(best_path, extra=extra)
            save_json(
                meta_path,
                {
                    "best_val_acc": float(best_val),
                    "model_path": str(best_path.as_posix()),
                    "num_classes": int(num_classes),
                    "input_dim": int(f),
                    "train_samples": int(len(train_idx)),
                    "val_samples": int(len(val_idx)),
                    "model_type": str(args.model_type),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                    "bidirectional": bool(args.bidirectional),
                },
            )

    print(f"Best val acc: {best_val:.4f}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()