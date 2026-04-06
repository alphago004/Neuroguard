"""
NEUROGUARD — Device DNA enrollment.

Enrollment converts a set of known-normal windows for a device into a
DeviceDNA: a compact statistical summary of that device's embedding
distribution. The scorer then measures how far a new window's embedding
sits from this distribution to produce an anomaly score.

What is DeviceDNA?
------------------
  centroid          : mean of all enrolled embeddings (shape 64,)
                      The "center of gravity" of normal behavior.
  sigma             : per-dimension std dev of enrolled embeddings
                      Used to build a Mahalanobis-style distance.
  threshold_distance: cosine distance above which we fire an ALERT.
                      Set at mean + k*std of enrollment distances, where
                      k=2.5 (captures 99%+ of normal under Gaussian
                      assumption with a comfortable buffer).
  n_windows         : number of windows used — needed to judge DNA quality.
                      DNA from 3 windows is far less reliable than 50.

Distance metric: cosine distance
---------------------------------
We use cosine distance (1 - cosine_similarity) rather than Euclidean
for scoring. Reason: during enrollment we L2-normalize all embeddings,
so every embedding lives on the unit hypersphere. Cosine distance on the
unit sphere is equivalent to angular distance — it is invariant to
embedding magnitude and focuses purely on direction.

Threshold calibration
---------------------
  1. Embed all normal windows → emb_i  (shape N×64, L2-normalized)
  2. Compute cosine distance from each emb_i to the centroid
  3. threshold = mean(distances) + k * std(distances), k=2.5
     This ensures < 1% false-positive rate under Gaussian assumption.
  4. Clip threshold to [MIN_THRESHOLD, MAX_THRESHOLD] to handle
     pathological cases (1-window enrollment, perfectly identical windows).

Persistence
-----------
DNA objects are pickled to data/processed/dna/<device_id>.pkl so the
system survives restarts without re-enrolling. The enrollment timestamp
is stored for drift monitoring (drift.py compares current centroid vs
enrolled centroid over time).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import pickle
from dataclasses import dataclass
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
from src.models.siamese import SiameseNetwork, build_model
from src.training.dataset import WindowRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM:   int   = 64
K_SIGMA:         float = 2.5   # threshold = mean_dist + K_SIGMA * std_dist
MIN_THRESHOLD:   float = 0.05  # never alert on tiny deviations (sensor noise)
MAX_THRESHOLD:   float = 0.95  # never require extreme anomaly to fire
MIN_WINDOWS_WARN: int  = 10    # warn if enrolling with fewer windows than this

DNA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dna"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# DeviceDNA dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeviceDNA:
    """Statistical fingerprint of one IoT device's normal behavior.

    Attributes:
        device_id:          Source IP or logical device name.
        centroid:           L2-normalized mean embedding (64,).
        sigma:              Per-dimension std dev of embeddings (64,).
                            Used for Mahalanobis fallback in drift.py.
        threshold_distance: Cosine distance above which scorer fires ALERT.
        n_windows:          Number of normal windows used for enrollment.
        enrolled_at:        UTC timestamp of enrollment.
        embedding_distances: Sorted cosine distances of enrolled windows
                             from centroid — used for threshold visualization.
    """
    device_id:           str
    centroid:            np.ndarray      # (64,) L2-normalized
    sigma:               np.ndarray      # (64,)
    threshold_distance:  float
    n_windows:           int
    enrolled_at:         datetime
    embedding_distances: np.ndarray      # (n_windows,) sorted ascending

    def __repr__(self) -> str:
        return (
            f"DeviceDNA(device_id={self.device_id!r}, "
            f"n_windows={self.n_windows}, "
            f"threshold={self.threshold_distance:.4f}, "
            f"enrolled_at={self.enrolled_at.strftime('%Y-%m-%d %H:%M')})"
        )


# ---------------------------------------------------------------------------
# Model loader (cached — loaded once per process)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Optional[tuple[SiameseNetwork, torch.device]] = None


def _load_model(
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
    transformer_layers: int = 1,
    margin: float = 3.0,
) -> tuple[SiameseNetwork, torch.device]:
    """Load the trained SiameseNetwork, caching after first call."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, _ = build_model(transformer_layers=transformer_layers, margin=margin)
    model = model.to(device)
    model.load(checkpoint_path, device=device)
    model.eval()

    _MODEL_CACHE = (model, device)
    logger.info(f"Model loaded from {checkpoint_path} on {device}")
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_windows(
    records: list[WindowRecord],
    model: SiameseNetwork,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Embed a list of WindowRecords into L2-normalized float32 vectors.

    Args:
        records:    Windows with pre-scaled features (RobustScaler applied).
        model:      Trained SiameseNetwork in eval mode.
        device:     Inference device.
        batch_size: Embedding batch size.

    Returns:
        numpy array of shape (N, 64), dtype float32, each row unit-norm.
    """
    all_features = np.stack([r.features for r in records])  # (N, 60)
    embeddings = []

    for start in range(0, len(all_features), batch_size):
        batch = torch.from_numpy(
            all_features[start:start + batch_size]
        ).to(device)
        emb = model.encoder.encode(batch, normalize=True)  # L2-normalized
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)   # (N, 64)


def embed_features(
    features: np.ndarray,
    model: SiameseNetwork,
    device: torch.device,
) -> np.ndarray:
    """Embed a single pre-scaled feature vector (60,) → (64,) L2-normalized.

    Args:
        features: float32 array of shape (60,).
        model:    Trained SiameseNetwork in eval mode.
        device:   Inference device.

    Returns:
        float32 array of shape (64,), L2-normalized.
    """
    with torch.no_grad():
        t = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, 60)
        emb = model.encoder.encode(t, normalize=True)            # (1, 64)
    return emb.cpu().numpy()[0]                                  # (64,)


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def enroll_device(
    device_id: str,
    normal_records: list[WindowRecord],
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
    k_sigma: float = K_SIGMA,
    save: bool = True,
) -> DeviceDNA:
    """Create a DeviceDNA fingerprint from known-normal windows.

    This is the ONLY function that should be called with normal traffic
    data. Attack windows must never be passed here.

    Algorithm
    ---------
    1. Embed all normal windows → emb_i  (N × 64, L2-normalized)
    2. centroid = mean(emb_i), then L2-normalize centroid
    3. sigma = std(emb_i, axis=0)
    4. dist_i = cosine_distance(emb_i, centroid)  for each window
    5. threshold = mean(dist_i) + k_sigma * std(dist_i)
    6. Clip threshold to [MIN_THRESHOLD, MAX_THRESHOLD]

    Args:
        device_id:       Device identifier string (e.g. '192.168.1.152').
        normal_records:  WindowRecords with label=0 and SCALED features.
        checkpoint_path: Path to best_model.pt.
        k_sigma:         Threshold sensitivity multiplier (default: 2.5).
        save:            If True, pickle DNA to DNA_DIR/<device_id>.pkl.

    Returns:
        DeviceDNA instance.

    Raises:
        ValueError: If normal_records is empty.
    """
    if not normal_records:
        raise ValueError(f"Cannot enroll {device_id}: no normal records provided")

    if len(normal_records) < MIN_WINDOWS_WARN:
        logger.warning(
            f"Device {device_id}: enrolling with only {len(normal_records)} windows. "
            f"DNA quality may be poor — recommend >= {MIN_WINDOWS_WARN} windows."
        )

    model, device = _load_model(checkpoint_path)

    # Step 1: embed all normal windows
    embeddings = embed_windows(normal_records, model, device)   # (N, 64)

    # Step 2: centroid — mean of L2-normalized embeddings, re-normalized
    centroid_raw = embeddings.mean(axis=0)                      # (64,)
    norm = np.linalg.norm(centroid_raw)
    centroid = centroid_raw / norm if norm > 1e-8 else centroid_raw

    # Step 3: per-dimension std dev (used by drift detector)
    sigma = embeddings.std(axis=0)                              # (64,)

    # Step 4: cosine distances from each embedding to centroid
    # cosine_distance = 1 - dot(emb, centroid)  (both are unit vectors)
    dot_products = embeddings @ centroid                        # (N,)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    distances = 1.0 - dot_products                             # (N,)

    # Step 5-6: threshold calibration
    mean_dist = float(distances.mean())
    std_dist  = float(distances.std())
    threshold = mean_dist + k_sigma * std_dist
    threshold = float(np.clip(threshold, MIN_THRESHOLD, MAX_THRESHOLD))

    dna = DeviceDNA(
        device_id=device_id,
        centroid=centroid.astype(np.float32),
        sigma=sigma.astype(np.float32),
        threshold_distance=threshold,
        n_windows=len(normal_records),
        enrolled_at=datetime.utcnow(),
        embedding_distances=np.sort(distances).astype(np.float32),
    )

    logger.info(
        f"Enrolled {device_id}: {len(normal_records)} windows | "
        f"mean_dist={mean_dist:.4f} | std={std_dist:.4f} | "
        f"threshold={threshold:.4f}"
    )

    if save:
        DNA_DIR.mkdir(parents=True, exist_ok=True)
        pkl_path = DNA_DIR / f"{device_id.replace('.', '_')}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(dna, f)
        logger.debug(f"DNA saved → {pkl_path}")

    return dna


def enroll_all_devices(
    window_dataset,            # WindowDataset — accepts type annotation at runtime
    scaler,                    # fitted RobustScaler
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
) -> dict[str, DeviceDNA]:
    """Enroll all devices using their test_normal (held-out) windows.

    Uses test_normal (not train_normal) for enrollment, matching the
    evaluation protocol in CLAUDE.md §11:
      Step 5 — Enroll devices using test_normal (held-out normal windows)

    Args:
        window_dataset:  WindowDataset with test_normal populated.
        scaler:          Fitted RobustScaler (from training).
        checkpoint_path: Path to best_model.pt.

    Returns:
        Dict mapping device_id → DeviceDNA.
    """
    from src.training.dataset import WindowRecord, LABEL_NORMAL
    from sklearn.preprocessing import RobustScaler

    # Scale test_normal features
    records = window_dataset.test_normal
    features_raw = np.stack([r.features for r in records])
    features_scaled = scaler.transform(features_raw).astype(np.float32)

    scaled_records = [
        WindowRecord(
            device_id=r.device_id,
            features=features_scaled[i],
            label=r.label,
            window_idx=r.window_idx,
            flow_start=r.flow_start,
        )
        for i, r in enumerate(records)
    ]

    # Group by device
    per_device: dict[str, list[WindowRecord]] = {}
    for r in scaled_records:
        per_device.setdefault(r.device_id, []).append(r)

    dna_map: dict[str, DeviceDNA] = {}
    for device_id, dev_records in sorted(per_device.items()):
        normal_only = [r for r in dev_records if r.label == LABEL_NORMAL]
        if not normal_only:
            logger.warning(f"Device {device_id}: no normal test windows — skipping enrollment")
            continue
        dna_map[device_id] = enroll_device(
            device_id=device_id,
            normal_records=normal_only,
            checkpoint_path=checkpoint_path,
        )

    logger.info(f"Enrolled {len(dna_map)} devices")
    return dna_map


def load_dna(device_id: str, dna_dir: Path = DNA_DIR) -> DeviceDNA:
    """Load a previously saved DeviceDNA from disk.

    Args:
        device_id: Device identifier string.
        dna_dir:   Directory containing pickled DNA files.

    Returns:
        DeviceDNA instance.

    Raises:
        FileNotFoundError: If DNA file does not exist.
    """
    pkl_path = dna_dir / f"{device_id.replace('.', '_')}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No DNA found for device {device_id!r} at {pkl_path}. "
            f"Run enroll_device() first."
        )
    with open(pkl_path, "rb") as f:
        dna = pickle.load(f)
    return dna
