"""Baselines: random/majority tags, CNN on mel-spectrogram."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MajorityBaseline:
    def fit(self, y: np.ndarray) -> "MajorityBaseline":
        # y: [N] multiclass or [N,K] multilabel
        if y.ndim == 1:
            vals, counts = np.unique(y, return_counts=True)
            self.mode = int(vals[np.argmax(counts)])
            self.multilabel = False
        else:
            self.prior = (y.mean(axis=0) >= 0.5).astype(np.float32)
            self.multilabel = True
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.multilabel:
            return np.tile(self.prior, (n, 1))
        return np.full(n, self.mode, dtype=np.int64)


class RandomBaseline:
    def __init__(self, num_classes: int, multilabel: bool = False, seed: int = 42):
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.rng = np.random.default_rng(seed)

    def predict(self, n: int) -> np.ndarray:
        if self.multilabel:
            return self.rng.random((n, self.num_classes)).astype(np.float32)
        return self.rng.integers(0, self.num_classes, size=n)


class MelCNN(nn.Module):
    """Small CNN on log-mel spectrogram for genre classification."""

    def __init__(self, n_mels: int = 128, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        # mel: [B, n_mels, T] or [B, 1, n_mels, T]
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        h = self.features(mel)
        logits = self.classifier(h)
        return {"logits": logits}
