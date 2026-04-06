"""
NEUROGUARD — Behavioral encoder (one twin of the Siamese network).

Architecture (from CLAUDE.md §4, locked)
-----------------------------------------
Input:  (batch, 60)  — one 50-flow window feature vector per sample
  → Linear(60  → 128)  + BatchNorm1d(128)  + ReLU
  → Linear(128 → 256)  + BatchNorm1d(256)  + ReLU + Dropout(0.3)
  → TemporalEncoder(d_model=256, nhead=4, num_layers=2)
  → Linear(256 → 128)  + ReLU
  → Linear(128 → 64)
Output: (batch, 64)  — behavioral embedding (L2-normalized at inference)

Design rationale
----------------
- BatchNorm1d after each Linear: stabilizes training on heterogeneous IoT
  features (byte counts span 0–10^5, ratios span 0–1). Without BN the
  gradient signal from ratio features is drowned out by raw byte counts.

- Dropout(0.3) after the 256-dim layer only: the first layer maps from
  raw features — full connectivity needed. The 256→128 projection and
  final 128→64 bottleneck are kept clean so the embedding space is smooth.

- No activation after the final Linear(128→64): the contrastive loss
  operates on Euclidean distances in the embedding space — an activation
  would constrain the embedding geometry unnecessarily. L2 normalization
  is applied at inference time in scorer.py, not here, so training
  distances are in raw Euclidean space (matching the ContrastiveLoss margin).

- TemporalEncoder uses Pre-LN (norm_first=True): more stable gradient
  flow through deep Transformer stacks (Xiong et al. 2020).

Weight initialization
---------------------
Linear layers use Kaiming uniform (PyTorch default for Linear) which is
correct for ReLU activations. BatchNorm is initialized to weight=1,
bias=0 (PyTorch default). No custom init needed — defaults are sound.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.models.transformer import TemporalEncoder
from src.features.extractor import FEATURE_DIM

# ---------------------------------------------------------------------------
# Constants (mirror of CLAUDE.md §10)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIM:    int   = FEATURE_DIM   # 60
DEFAULT_HIDDEN_1:     int   = 128
DEFAULT_HIDDEN_2:     int   = 256
DEFAULT_EMBEDDING_DIM: int  = 64
DEFAULT_DROPOUT:      float = 0.3
DEFAULT_NHEAD:        int   = 4
DEFAULT_TRANSFORMER_LAYERS: int = 2


class BehavioralEncoder(nn.Module):
    """Maps a 60-dim flow-window feature vector to a 64-dim behavioral embedding.

    The same encoder instance is used by both twins of the Siamese network
    (shared weights). This is the critical design choice: shared weights
    mean both twins learn the same notion of "device identity" — the
    embedding space is jointly optimized so that same-device windows
    cluster together and different-device windows separate.

    Args:
        input_dim:      Input feature dimension (default: 60).
        embedding_dim:  Output embedding dimension (default: 64).
        dropout:        Dropout rate after the 256-dim layer (default: 0.3).
        nhead:          Transformer attention heads (default: 4).
        transformer_layers: Transformer depth (default: 2).

    Example:
        >>> encoder = BehavioralEncoder()
        >>> x = torch.randn(8, 60)
        >>> embeddings = encoder(x)
        >>> embeddings.shape
        torch.Size([8, 64])
    """

    def __init__(
        self,
        input_dim:          int   = DEFAULT_INPUT_DIM,
        embedding_dim:      int   = DEFAULT_EMBEDDING_DIM,
        dropout:            float = DEFAULT_DROPOUT,
        nhead:              int   = DEFAULT_NHEAD,
        transformer_layers: int   = DEFAULT_TRANSFORMER_LAYERS,
    ) -> None:
        super().__init__()

        # ── Layer 1: input projection ──────────────────────────────────────
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, DEFAULT_HIDDEN_1, bias=False),  # bias=False: BN has its own bias
            nn.BatchNorm1d(DEFAULT_HIDDEN_1),
            nn.ReLU(inplace=True),
        )

        # ── Layer 2: feature expansion + regularization ───────────────────
        self.layer2 = nn.Sequential(
            nn.Linear(DEFAULT_HIDDEN_1, DEFAULT_HIDDEN_2, bias=False),
            nn.BatchNorm1d(DEFAULT_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Layer 3: Transformer refinement ───────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            d_model=DEFAULT_HIDDEN_2,
            nhead=nhead,
            num_layers=transformer_layers,
            dropout=0.1,              # light dropout inside Transformer
            dim_feedforward=DEFAULT_HIDDEN_2 * 2,  # 512
        )

        # ── Layer 4: embedding projection ─────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(DEFAULT_HIDDEN_2, DEFAULT_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.Linear(DEFAULT_HIDDEN_1, embedding_dim),
            # No activation — raw embedding space for contrastive loss
        )

        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim

        # Log parameter count once at construction
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.debug(
            f"BehavioralEncoder: {input_dim}→{embedding_dim} | "
            f"{n_params:,} trainable parameters"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of feature vectors into behavioral embeddings.

        Args:
            x: Float32 tensor of shape (batch, input_dim).

        Returns:
            Float32 tensor of shape (batch, embedding_dim).
        """
        x = self.layer1(x)             # (batch, 128)
        x = self.layer2(x)             # (batch, 256)
        x = self.temporal_encoder(x)   # (batch, 256)
        x = self.projection(x)         # (batch, 64)
        return x

    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Inference-time encoding with optional L2 normalization.

        L2 normalization maps all embeddings onto the unit hypersphere,
        which is required for cosine-distance comparison in scorer.py.
        We do NOT normalize during training — the ContrastiveLoss uses
        raw Euclidean distance, and normalizing during training would
        conflict with the margin geometry.

        Args:
            x:         Input tensor (batch, input_dim).
            normalize: If True, L2-normalize the output (default: True).

        Returns:
            Embedding tensor (batch, embedding_dim), optionally normalized.
        """
        with torch.no_grad():
            emb = self.forward(x)
            if normalize:
                emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
