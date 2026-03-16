from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class SequenceRNNConfig:
    input_dim: int
    num_classes: int
    model_type: str = "gru"   # "gru" or "lstm"
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False
    seed: int = 42


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


class SequenceRNNClassifier(nn.Module):
    """
    Temporal sequence classifier for ASL landmarks.

    Input:
        X: (N, T, F)
        mask: (N, T) with 1=valid, 0=pad

    Model:
        sequence -> GRU/LSTM -> last valid hidden state -> linear
    """

    def __init__(self, config: SequenceRNNConfig):
        super().__init__()
        self.config = config

        torch.manual_seed(config.seed)

        model_type = config.model_type.lower()
        if model_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM

        self.rnn = rnn_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(out_dim, config.num_classes)

    def forward_torch(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Torch forward for training.
        X: (N, T, F)
        mask: (N, T)
        returns: logits (N, C)
        """
        if X.ndim != 3:
            raise ValueError(f"X must have shape (N,T,F), got {tuple(X.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (N,T), got {tuple(mask.shape)}")
        if X.shape[:2] != mask.shape:
            raise ValueError(f"X/mask mismatch: X={tuple(X.shape)} mask={tuple(mask.shape)}")

        lengths = mask.sum(dim=1).long().clamp(min=1)

        packed = pack_padded_sequence(
            X,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed)

        # LSTM returns (h_n, c_n)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.config.bidirectional:
            # last layer forward + backward
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

    def forward(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Numpy friendly forward for inference compatibility.
        Returns logits as numpy array.
        """
        device = next(self.parameters()).device
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)

        self.eval()
        with torch.no_grad():
            logits = self.forward_torch(X_t, mask_t)

        return logits.detach().cpu().numpy().astype(np.float32)

    def predict(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = self.forward(X, mask)
        return logits.argmax(axis=1).astype(np.int64)

    def save(self, path: str | Path, extra: Dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "type": "SequenceRNNClassifier",
            "config": {
                "input_dim": int(self.config.input_dim),
                "num_classes": int(self.config.num_classes),
                "model_type": str(self.config.model_type),
                "hidden_dim": int(self.config.hidden_dim),
                "num_layers": int(self.config.num_layers),
                "dropout": float(self.config.dropout),
                "bidirectional": bool(self.config.bidirectional),
                "seed": int(self.config.seed),
            },
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }

        torch.save(payload, path)

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> "SequenceRNNClassifier":
        path = Path(path)
        payload = torch.load(path, map_location=map_location)

        cfg = payload["config"]
        config = SequenceRNNConfig(
            input_dim=int(cfg["input_dim"]),
            num_classes=int(cfg["num_classes"]),
            model_type=str(cfg.get("model_type", "gru")),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
            bidirectional=bool(cfg.get("bidirectional", False)),
            seed=int(cfg.get("seed", 42)),
        )

        model = SequenceRNNClassifier(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)