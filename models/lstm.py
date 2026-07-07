"""Bidirectional LSTM with attention for ICU mortality prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: (B, T, H)
        scores = self.attn(hidden).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=-1)            # (B, T)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, H)
        return context, weights


class LSTMMortality(nn.Module):
    """
    Bidirectional LSTM → temporal attention → classifier.

    Parameters
    ----------
    input_dim   : number of vital-sign features (default 5)
    hidden_dim  : LSTM units per direction
    num_layers  : stacked LSTM layers
    dropout     : applied between LSTM layers and before classifier
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)  # bi-directional
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        x = F.relu(self.input_proj(x))         # (B, T, H)
        out, _ = self.lstm(x)                   # (B, T, 2H)
        context, attn_weights = self.attention(out)  # (B, 2H), (B, T)
        context = self.dropout(context)
        logits = self.classifier(context).squeeze(-1)  # (B,)
        return logits, attn_weights
