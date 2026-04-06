"""
NEUROGUARD — Real-time anomaly scorer.

Takes a single behavioral window (feature vector), embeds it using the
trained encoder, measures its cosine distance from the device's DNA
centroid, and returns an AnomalyResult.

Anomaly score definition
------------------------
  raw_distance = cosine_distance(embedding, dna.centroid)
               = 1 - dot(L2_norm(embedding), dna.centroid)

  anomaly_score = raw_distance / dna.threshold_distance

  This normalizes the score so that:
    score = 1.0 → exactly at the alert threshold
    score < 1.0 → normal behavior (lower = more normal)
    score > 1.0 → anomalous (higher = more deviant)

  A score of 0.0 means the window is identical to the DNA centroid
  (theoretically impossible in practice, but well-defined).

Why cosine distance?
---------------------
After L2 normalization all embeddings live on the unit hypersphere.
Cosine distance measures the angular separation between a new window's
embedding and the device's centroid embedding. It is:
  - Scale-invariant: amplitude of activation doesn't matter, only direction
  - Bounded: [0, 2] (0 = same direction, 2 = opposite direction)
  - Well-suited for high-dimensional spaces (less affected by curse of
    dimensionality than Euclidean distance)

SHAP attribution (ALERT only)
-------------------------------
When an ALERT fires, we compute a lightweight gradient-based attribution
to identify which of the 60 input features contributed most to the
deviation. This uses the gradient of the embedding distance w.r.t. the
input features — a first-order Taylor approximation that is fast enough
for real-time use without requiring the full SHAP TreeExplainer.

Full SHAP (DeepExplainer) is used during offline evaluation in
src/training/metrics.py for paper-quality feature importance analysis.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.detection.enroll import (
    DeviceDNA,
    embed_features,
    _load_model,
    CHECKPOINT_DIR,
)
from src.features.extractor import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATUS_NORMAL = "NORMAL"
STATUS_ALERT  = "ALERT"
TOP_N_FEATURES = 3   # number of features to report in AnomalyResult


# ---------------------------------------------------------------------------
# AnomalyResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Result of scoring one behavioral window against a device's DNA.

    Attributes:
        device_id:     Device identifier.
        anomaly_score: Normalized distance score.
                       < 1.0 → NORMAL, ≥ 1.0 → ALERT.
        raw_distance:  Raw cosine distance (before threshold normalization).
        threshold:     The device's alert threshold (from DeviceDNA).
        status:        'NORMAL' or 'ALERT'.
        top_features:  Top-3 contributing feature names (populated on ALERT).
        timestamp:     UTC time of scoring.
    """
    device_id:     str
    anomaly_score: float
    raw_distance:  float
    threshold:     float
    status:        str
    top_features:  list[str] = field(default_factory=list)
    timestamp:     datetime  = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        feat_str = f", top_features={self.top_features}" if self.top_features else ""
        return (
            f"AnomalyResult(device={self.device_id!r}, "
            f"score={self.anomaly_score:.4f}, "
            f"status={self.status!r}"
            f"{feat_str})"
        )


# ---------------------------------------------------------------------------
# Gradient-based feature attribution
# ---------------------------------------------------------------------------

def _gradient_attribution(
    features: np.ndarray,
    dna: DeviceDNA,
    model,
    device: torch.device,
    top_n: int = TOP_N_FEATURES,
) -> list[str]:
    """Return top-N feature names by gradient magnitude w.r.t. cosine distance.

    Computes d(cosine_distance) / d(input_features) via autograd.
    The absolute gradient value reflects how sensitive the anomaly score
    is to each feature — the features with largest |grad| are the ones
    most responsible for the deviation.

    This is intentionally lightweight (single forward+backward pass).
    For publication-quality attribution, use SHAP DeepExplainer offline.

    Args:
        features: Pre-scaled feature vector, shape (60,).
        dna:      DeviceDNA for the device.
        model:    Trained SiameseNetwork in eval mode.
        device:   Inference device.
        top_n:    Number of top features to return.

    Returns:
        List of top_n feature name strings.
    """
    model.encoder.eval()
    t = torch.from_numpy(features).unsqueeze(0).to(device).requires_grad_(True)

    # Forward pass through encoder (no normalize here — need raw embedding for grad)
    emb = model.encoder(t)                           # (1, 64)
    emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)  # (1, 64)

    # Cosine distance = 1 - dot(emb_norm, centroid)
    centroid_t = torch.from_numpy(dna.centroid).to(device)      # (64,)
    cos_sim = (emb_norm[0] * centroid_t).sum()
    cos_dist = 1.0 - cos_sim

    cos_dist.backward()

    grad = t.grad[0].cpu().numpy()          # (60,)
    abs_grad = np.abs(grad)
    top_indices = np.argsort(abs_grad)[::-1][:top_n]
    return [FEATURE_NAMES[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_window(
    device_id: str,
    features: np.ndarray,
    dna: DeviceDNA,
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
    compute_attribution: bool = True,
) -> AnomalyResult:
    """Score a single pre-scaled feature window against a device's DNA.

    Args:
        device_id:            Device identifier (must match dna.device_id).
        features:             Pre-scaled float32 array, shape (60,).
        dna:                  Enrolled DeviceDNA for this device.
        checkpoint_path:      Path to best_model.pt.
        compute_attribution:  If True, compute top-feature attribution on ALERT.

    Returns:
        AnomalyResult with score, status, and (on ALERT) top_features.
    """
    if features.shape != (60,):
        raise ValueError(f"Expected features shape (60,), got {features.shape}")

    model, device = _load_model(checkpoint_path)

    # Embed and L2-normalize
    embedding = embed_features(features, model, device)         # (64,) unit-norm

    # Cosine distance to centroid
    dot = float(np.clip(np.dot(embedding, dna.centroid), -1.0, 1.0))
    raw_distance = 1.0 - dot

    # Normalize by threshold → anomaly score
    anomaly_score = raw_distance / dna.threshold_distance

    is_alert = anomaly_score >= 1.0
    status = STATUS_ALERT if is_alert else STATUS_NORMAL

    top_features: list[str] = []
    if is_alert and compute_attribution:
        try:
            top_features = _gradient_attribution(features, dna, model, device)
        except Exception as exc:
            logger.warning(f"Attribution failed for {device_id}: {exc}")

    return AnomalyResult(
        device_id=device_id,
        anomaly_score=float(anomaly_score),
        raw_distance=float(raw_distance),
        threshold=dna.threshold_distance,
        status=status,
        top_features=top_features,
    )


# ---------------------------------------------------------------------------
# Batch scoring (convenience for evaluation)
# ---------------------------------------------------------------------------

def score_records(
    records: list,              # list[WindowRecord] with scaled features
    dna_map: dict[str, DeviceDNA],
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
    compute_attribution: bool = False,  # off by default for batch speed
) -> list[AnomalyResult]:
    """Score a list of WindowRecords against their enrolled device DNAs.

    Windows belonging to devices not in dna_map are skipped.

    Args:
        records:              List of WindowRecord (scaled features).
        dna_map:              Dict mapping device_id → DeviceDNA.
        checkpoint_path:      Path to best_model.pt.
        compute_attribution:  Whether to run gradient attribution on ALERTs.

    Returns:
        List of AnomalyResult, one per scored record.
    """
    results: list[AnomalyResult] = []
    skipped = 0

    for record in records:
        dna = dna_map.get(record.device_id)
        if dna is None:
            skipped += 1
            continue
        result = score_window(
            device_id=record.device_id,
            features=record.features,
            dna=dna,
            checkpoint_path=checkpoint_path,
            compute_attribution=compute_attribution,
        )
        results.append(result)

    if skipped:
        logger.warning(f"Skipped {skipped} records with no enrolled DNA")

    return results
