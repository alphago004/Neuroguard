"""
NEUROGUARD — Temporal encoder block (Transformer).

Wraps PyTorch's built-in TransformerEncoder and handles the awkward
shape mismatch between our flat feature vectors and the sequence-first
convention that nn.TransformerEncoder expects.

Why a Transformer here?
-----------------------
IoT devices have *temporal rhythm* — a thermostat polls at fixed
intervals, a GPS beacon fires every N seconds, a modbus device has
strict request-response cadence. A Transformer's self-attention can
capture which positions in the 50-flow window are unusual relative to
the rest of the window. This is the "temporal rhythm capture" claim
in the paper.

The encoder receives a (batch, d_model) tensor — one embedding per
window (not a time series of tokens). We treat each window as a
single-token sequence for the Transformer, which means self-attention
operates across the batch but not across time. This is intentional:
we're asking the Transformer to refine the per-window representation
using position-independent attention over the feature dimensions.

For the full temporal extension (feeding a sequence of consecutive
windows as tokens), see Section 14 of CLAUDE.md — this is tracked
as a future paper contribution.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from loguru import logger


class TemporalEncoder(nn.Module):
    """Transformer-based refinement block for behavioral embeddings.

    Takes a (batch, d_model) tensor, reshapes to (1, batch, d_model) to
    satisfy the (seq_len, batch, d_model) convention of nn.TransformerEncoder,
    runs multi-head self-attention, and returns (batch, d_model).

    Args:
        d_model:    Embedding dimension — must match encoder hidden dim (256).
        nhead:      Number of attention heads (default: 4).
                    d_model must be divisible by nhead.
        num_layers: Number of stacked TransformerEncoderLayer blocks (default: 2).
        dropout:    Dropout applied inside each TransformerEncoderLayer (default: 0.1).
                    Kept low here — the BehavioralEncoder already applies Dropout(0.3).
        dim_feedforward: Inner dimension of the FFN inside each layer (default: 512 = 2×d_model).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
                f"Current ratio: {d_model / nhead:.2f}"
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,   # expects (seq, batch, d_model)
            norm_first=True,     # Pre-LN: more stable training than post-LN
                                 # (Xiong et al. 2020 "On Layer Normalization in
                                 # the Transformer Architecture")
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoid MPS incompatibility warning
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a batch of embeddings via Transformer self-attention.

        Args:
            x: Tensor of shape (batch, d_model).

        Returns:
            Tensor of shape (batch, d_model) — same shape as input.
        """
        # nn.TransformerEncoder expects (seq_len, batch, d_model)
        # We treat the whole batch as a single sequence of length 1
        # so each sample attends only to itself — pure per-sample refinement.
        # Shape: (batch, d_model) → (1, batch, d_model)
        x = x.unsqueeze(0)
        x = self.transformer(x)     # → (1, batch, d_model)
        x = x.squeeze(0)            # → (batch, d_model)
        return x
