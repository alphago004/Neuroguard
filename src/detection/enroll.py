"""
NEUROGUARD — Device DNA enrollment.

Converts known-normal windows for a device into a DeviceDNA: the mean
embedding (centroid), per-dimension std, and a cosine-distance threshold
calibrated at mean + 2.5σ of enrollment distances.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Statistical fingerprint of one IoT device's normal behavior."""
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

    Embeds all records, computes centroid and per-dim sigma, then sets
    threshold = mean(cosine_dist) + k_sigma * std(cosine_dist).

    Args:
        device_id:       Device identifier (e.g. '192.168.1.152').
        normal_records:  WindowRecords with label=0 and SCALED features.
        checkpoint_path: Path to best_model.pt.
        k_sigma:         Threshold multiplier (default: 2.5).
        save:            Pickle DNA to DNA_DIR/<device_id>.pkl if True.
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
        enrolled_at=datetime.now(timezone.utc),
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
    """Enroll all devices using their enroll_normal windows and return a device_id → DeviceDNA map."""
    from src.training.dataset import WindowRecord, LABEL_NORMAL

    # Scale enroll_normal features
    records = window_dataset.enroll_normal
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
            logger.warning(f"Device {device_id}: no enroll_normal windows — skipping enrollment")
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
