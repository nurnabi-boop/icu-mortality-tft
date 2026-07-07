"""
Temporal Fusion Transformer (Lim et al., 2021) for ICU mortality prediction.

Full architecture implemented from scratch:
  - Gated Residual Networks (GRN)
  - Variable Selection Networks (VSN)
  - LSTM encoder (temporal processing)
  - Multi-head self-attention with interpretable attention
  - Gated skip connections at every stage
  - Point-wise feed-forward output layer → binary classifier

Reference: https://arxiv.org/abs/1912.09363
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit: splits the last dimension in half, gates one half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """
    Gated Residual Network.

        GRN(x, c=None) = LayerNorm(x + GLU( W2·ELU(W1·[x; c]) ))

    Parameters
    ----------
    input_dim   : dimension of x
    hidden_dim  : internal projection dimension
    output_dim  : output dimension (if None, equals input_dim)
    context_dim : dimension of optional context c (0 to skip)
    dropout     : applied before the GLU gate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.w1 = nn.Linear(input_dim + context_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim * 2)  # ×2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

        # Skip connection projection when dimensions differ
        self.skip = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
        self.glu = GLU()

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([x, c], dim=-1) if c is not None else x
        h = F.elu(self.w1(inp))
        h = self.dropout(h)
        h = self.glu(self.w2(h))
        return self.norm(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """
    Selects and transforms one feature variable at a time, then softmax-weights.

    Parameters
    ----------
    num_features : number of input features (F)
    input_dim    : dimensionality of each feature after initial projection (d_model)
    hidden_dim   : GRN hidden size
    context_dim  : optional static context dimension
    dropout      : dropout rate
    """

    def __init__(
        self,
        num_features: int,
        input_dim: int,
        hidden_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GRN(input_dim, hidden_dim, input_dim, context_dim=context_dim, dropout=dropout)
            for _ in range(num_features)
        ])

        # Flat selection GRN → softmax weights
        self.flat_grn = GRN(
            num_features * input_dim,
            hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, T, F, d_model) or (B, F, d_model)
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        processed : (B, T, d_model) or (B, d_model) — weighted sum
        weights   : (B, T, F)      or (B, F)         — selection weights
        """
        has_time = x.dim() == 4
        if not has_time:
            x = x.unsqueeze(1)   # treat as T=1

        B, T, nF, D = x.shape

        # Per-variable processing
        var_outputs = []
        for i in range(nF):
            xi = x[:, :, i, :]                              # (B, T, D)
            ci = c.unsqueeze(1).expand(B, T, -1) if c is not None else None
            var_outputs.append(self.var_grns[i](xi, ci))    # (B, T, D)
        var_stack = torch.stack(var_outputs, dim=2)          # (B, T, nF, D)

        # Flat weights
        flat = x.reshape(B, T, nF * D)                      # (B, T, nF*D)
        ci_flat = c.unsqueeze(1).expand(B, T, -1) if c is not None else None
        weights = torch.softmax(self.flat_grn(flat, ci_flat), dim=-1)  # (B, T, nF)

        processed = (weights.unsqueeze(-1) * var_stack).sum(dim=2)  # (B, T, D)

        if not has_time:
            processed = processed.squeeze(1)
            weights   = weights.squeeze(1)

        return processed, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention where head outputs share a single value projection,
    enabling interpretable per-head weight averaging (Lim et al. §3.4).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, self.d_head, bias=False)  # shared
        self.W_o  = nn.Linear(self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape
        H, D_h = self.num_heads, self.d_head

        # Project Q, K per head; shared V
        Q = self.W_q(q).view(B, T, H, D_h).transpose(1, 2)   # (B, H, T, D_h)
        K = self.W_k(k).view(B, T, H, D_h).transpose(1, 2)
        V = self.W_v(v)                                         # (B, T, D_h)

        scale = math.sqrt(D_h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # Expand V for each head then aggregate
        V_exp = V.unsqueeze(1).expand(B, H, T, D_h)
        head_out = torch.matmul(attn, V_exp)                   # (B, H, T, D_h)
        out = head_out.mean(dim=1)                             # (B, T, D_h) — interpretable
        out = self.W_o(out)                                     # (B, T, d_model)

        attn_avg = attn.mean(dim=1)                            # (B, T, T)
        return out, attn_avg


# ---------------------------------------------------------------------------
# Full TFT
# ---------------------------------------------------------------------------

class TFTMortality(nn.Module):
    """
    Temporal Fusion Transformer adapted for binary classification.

    Treats all 5 vitals as past time-varying inputs (no static covariates).
    The 48-h series is encoded through the full TFT pipeline; the final
    time step's representation is used for mortality prediction.

    Parameters
    ----------
    input_dim   : number of input features (vitals), default 5
    d_model     : model / embedding dimension
    num_heads   : attention heads (d_model must be divisible)
    num_lstm_layers : LSTM encoder/decoder layers
    dropout     : regularisation rate
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model   = d_model

        # 1. Input feature embeddings (linear projection per feature)
        self.input_embeds = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])

        # 2. Variable selection for time-varying inputs
        self.vsn = VariableSelectionNetwork(
            num_features=input_dim,
            input_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout,
        )

        # 3. Sequence-to-sequence LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # 4. Gate after LSTM (GLU + skip + norm)
        self.lstm_gate = nn.Linear(d_model, d_model * 2)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.glu = GLU()

        # 5. Static enrichment GRN (trivial here — no static features)
        self.static_enrich_grn = GRN(d_model, d_model, dropout=dropout)

        # 6. Interpretable multi-head self-attention
        self.self_attn = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.attn_gate = nn.Linear(d_model, d_model * 2)
        self.attn_norm = nn.LayerNorm(d_model)

        # 7. Position-wise feed-forward (GRN)
        self.ff_grn  = GRN(d_model, d_model * 4, d_model, dropout=dropout)
        self.ff_gate = nn.Linear(d_model, d_model * 2)
        self.ff_norm = nn.LayerNorm(d_model)

        # 8. Binary classifier on the last time step
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    # ------------------------------------------------------------------
    def _gated_add_norm(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        gate_layer: nn.Linear,
        norm_layer: nn.LayerNorm,
    ) -> torch.Tensor:
        gated = self.glu(gate_layer(x))
        return norm_layer(skip + gated)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x : (B, T, F)

        Returns
        -------
        logits      : (B,)
        attn_info   : dict with variable_weights (B, T, F) and temporal_attn (B, T, T)
        """
        B, T, nF = x.shape

        # --- 1. Embed each feature separately → (B, T, nF, d_model)
        embeds = torch.stack(
            [self.input_embeds[i](x[:, :, i:i+1]) for i in range(nF)],
            dim=2,
        )

        # --- 2. Variable selection → (B, T, d_model)
        vsn_out, var_weights = self.vsn(embeds)   # (B,T,d), (B,T,F)

        # --- 3. LSTM encoder
        lstm_out, _ = self.lstm_encoder(vsn_out)  # (B, T, d_model)

        # --- 4. Gated residual around LSTM
        h = self._gated_add_norm(lstm_out, vsn_out, self.lstm_gate, self.lstm_norm)

        # --- 5. Static enrichment (identity — no static covariates)
        enriched = self.static_enrich_grn(h)      # (B, T, d_model)

        # --- 6. Interpretable self-attention
        sa_out, temporal_attn = self.self_attn(enriched, enriched, enriched)
        h2 = self._gated_add_norm(sa_out, enriched, self.attn_gate, self.attn_norm)

        # --- 7. Point-wise feed-forward
        ff_out = self.ff_grn(h2)
        h3 = self._gated_add_norm(ff_out, h2, self.ff_gate, self.ff_norm)

        # --- 8. Classify using last time step
        last = h3[:, -1, :]                       # (B, d_model)
        logits = self.classifier(last).squeeze(-1)  # (B,)

        attn_info = {
            "variable_weights": var_weights,        # (B, T, F)
            "temporal_attn":    temporal_attn,      # (B, T, T)
        }
        return logits, attn_info
