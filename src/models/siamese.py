"""
NEUROGUARD — Siamese network and contrastive loss.

The Siamese network is the architectural core of the zero-day detection
claim. Two copies of the BehavioralEncoder process two windows in parallel
using SHARED weights, then ContrastiveLoss penalizes:
  - Same-device pairs that are far apart   (false negatives in identity)
  - Different-device pairs that are close  (false positives in identity)

Contrastive Loss (Hadsell et al. 2006)
---------------------------------------
  L = (1 - y) · d²  +  y · max(0, margin - d)²

  where:
    d      = Euclidean distance between the two embeddings
    y      = 0 if same device (positive pair), 1 if different device (negative)
    margin = 3.0 (chosen empirically — see below)

  Intuition:
    y=0 (same device): loss = d² → push distance toward 0
    y=1 (diff device): loss = max(0, 3.0 - d)² → push distance above 3.0
                       once d ≥ margin the loss is zero (no wasted gradient)

Why margin=3.0?
  Our target metrics:
    intra-class distance < 0.5  (same device windows cluster tightly)
    inter-class distance > 1.5  (different devices clearly separated)
  With margin=2.0 (literature default), inter-class distances saturated near
  1.5 — the margin was hit too early and gradient flow diminished before
  sufficient separation was achieved.  Widening to margin=3.0 keeps negative
  pairs in the loss-active zone longer, producing the 3.08× separation ratio
  reported in the paper (vs ~2.83× at margin=2.0).

SiameseNetwork
--------------
  - Holds ONE encoder instance — both forward() calls go through the
    same weights. This is the definition of Siamese architecture.
  - Returns (emb_a, emb_b) — the caller (training loop) computes the
    distance and passes it to ContrastiveLoss.
  - Does NOT normalize embeddings during training (see encoder.py).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.models.encoder import BehavioralEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MARGIN: float = 3.0


# ---------------------------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """Hadsell et al. (2006) contrastive loss for Siamese training.

    Args:
        margin: Minimum desired distance between embeddings of different
                devices. Pairs already separated by more than margin
                contribute zero loss. (Default: 3.0)

    Example:
        >>> loss_fn = ContrastiveLoss(margin=3.0)
        >>> dist = torch.tensor([0.1, 1.8, 0.05])   # Euclidean distances
        >>> labels = torch.tensor([0.0, 1.0, 0.0])  # 0=same, 1=different
        >>> loss = loss_fn(dist, labels)
    """

    def __init__(self, margin: float = DEFAULT_MARGIN) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean contrastive loss over a batch.

        Args:
            distance: Euclidean distances (batch,) — non-negative float32.
            label:    Pair labels (batch,) — 0.0=same device, 1.0=different.

        Returns:
            Scalar mean loss over the batch.
        """
        # Positive loss: same-device pairs pulled together
        # L_pos = (1 - y) · d²
        loss_pos = (1.0 - label) * distance.pow(2)

        # Negative loss: different-device pairs pushed apart
        # L_neg = y · max(0, margin - d)²
        loss_neg = label * F.relu(self.margin - distance).pow(2)

        loss = 0.5 * (loss_pos + loss_neg)
        return loss.mean()

    def extra_repr(self) -> str:
        return f"margin={self.margin}"


# ---------------------------------------------------------------------------
# Siamese Network
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """Siamese twin network for behavioral fingerprint learning.

    Passes two windows through the SAME encoder (shared weights) and
    returns their embeddings. The training loop then computes pairwise
    Euclidean distance and calls ContrastiveLoss.

    Args:
        encoder: A BehavioralEncoder instance. Both twins share this object.

    Example:
        >>> encoder = BehavioralEncoder()
        >>> model = SiameseNetwork(encoder)
        >>> anchor = torch.randn(8, 60)
        >>> pair   = torch.randn(8, 60)
        >>> emb_a, emb_b = model(anchor, pair)
        >>> emb_a.shape  # → torch.Size([8, 64])
    """

    def __init__(self, encoder: BehavioralEncoder) -> None:
        super().__init__()
        self.encoder = encoder   # shared — intentionally ONE instance

    def forward(
        self,
        anchor: torch.Tensor,
        pair:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both windows using the shared encoder.

        Args:
            anchor: Float32 tensor (batch, input_dim) — reference window.
            pair:   Float32 tensor (batch, input_dim) — comparison window.

        Returns:
            Tuple (emb_anchor, emb_pair), each (batch, embedding_dim).
        """
        emb_anchor = self.encoder(anchor)
        emb_pair   = self.encoder(pair)
        return emb_anchor, emb_pair

    @staticmethod
    def euclidean_distance(
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute per-sample Euclidean distance between two embedding batches.

        Args:
            emb_a: (batch, dim)
            emb_b: (batch, dim)
            eps:   Small value added under sqrt to prevent zero-gradient at d=0.

        Returns:
            (batch,) distance tensor, non-negative float32.
        """
        return torch.sqrt(
            torch.sum((emb_a - emb_b).pow(2), dim=1) + eps
        )

    def save(self, path: Path) -> None:
        """Save model state dict to disk.

        Args:
            path: Destination .pt file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SiameseNetwork saved → {path} ({n_params:,} params)")

    def load(self, path: Path, device: Optional[torch.device] = None) -> None:
        """Load model state dict from disk.

        Args:
            path:   Source .pt file path.
            device: Target device (defaults to current model device).
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=device or next(self.parameters()).device)
        self.load_state_dict(state)
        logger.info(f"SiameseNetwork loaded ← {path}")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(
    input_dim:          int   = 60,
    embedding_dim:      int   = 64,
    dropout:            float = 0.3,
    nhead:              int   = 4,
    transformer_layers: int   = 1,
    margin:             float = DEFAULT_MARGIN,
) -> tuple[SiameseNetwork, ContrastiveLoss]:
    """Construct a SiameseNetwork + ContrastiveLoss pair from hyperparameters.

    This is the canonical way to create the model — the training loop,
    notebooks, and tests should all go through this function so there is
    a single source of truth for the architecture.

    Args:
        input_dim:          Feature vector size (default: 60).
        embedding_dim:      Output embedding size (default: 64).
        dropout:            Encoder dropout (default: 0.3).
        nhead:              Transformer attention heads (default: 4).
        transformer_layers: Transformer depth (default: 1).
        margin:             Contrastive loss margin (default: 3.0).

    Returns:
        (SiameseNetwork, ContrastiveLoss) — both in train mode.
    """
    encoder = BehavioralEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        nhead=nhead,
        transformer_layers=transformer_layers,
    )
    model    = SiameseNetwork(encoder)
    loss_fn  = ContrastiveLoss(margin=margin)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built: input={input_dim} → embedding={embedding_dim} | "
        f"margin={margin} | {n_params:,} trainable parameters"
    )
    return model, loss_fn
