"""
NEUROGUARD — Windowing engine and Siamese training pair dataset.

Loads train_test_network.csv, slides 50-flow windows over each IoT device's
flows, extracts 60-dim feature vectors, and builds (anchor, pair, label) tuples
for Siamese contrastive training.

Key classes: build_windows(), WindowDataset, PairDataset, WindowRecord.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.features.extractor import extract_features, FEATURE_DIM, WINDOW_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_STRIDE: int = 25           # stride between windows (50% overlap)
LOCAL_IP_PREFIX: str = "192.168.1."  # only this subnet = IoT devices
TRAIN_SPLIT:  float = 0.80        # 80% of normal windows → training (chronological)
# Remaining 20% is subdivided into:
ENROLL_FRAC:  float = 0.70        # 70% of remaining 20% → enroll (14% total), rest → test (6%)
DEVICE_SAMPLE_CAP: int = 100      # max windows sampled per device per epoch
                                  # prevents device .152 (874 windows) from
                                  # dominating pair distribution

# Window label constants
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowRecord:
    """One 50-flow behavioral window for a single device."""
    device_id:  str
    features:   np.ndarray
    label:      int
    window_idx: int
    flow_start: int


# ---------------------------------------------------------------------------
# Core windowing engine
# ---------------------------------------------------------------------------

def build_windows(
    csv_path: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    local_prefix: str = LOCAL_IP_PREFIX,
) -> "WindowDataset":
    """Load TON_IoT CSV and extract all behavioral windows for IoT devices.

    Args:
        csv_path:     Path to train_test_network.csv (utf-8-sig encoded).
        window_size:  Number of flows per window (default: 50).
        stride:       Step size between consecutive windows (default: 25).
        local_prefix: IP prefix that identifies IoT devices (default: '192.168.1.').

    Returns:
        WindowDataset containing all extracted windows.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(df):,} rows, {df['src_ip'].nunique()} unique src_ips")

    # Validate expected columns exist
    required = {"src_ip", "dst_ip", "proto", "type", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    # Isolate IoT device rows (local subnet only)
    iot_mask = df["src_ip"].str.startswith(local_prefix)
    df_iot = df[iot_mask].copy()
    logger.info(f"IoT device rows: {len(df_iot):,} ({iot_mask.sum() / len(df) * 100:.1f}% of dataset)")

    all_records: list[WindowRecord] = []

    device_ips = df_iot["src_ip"].unique()
    for device_id in sorted(device_ips):
        device_df = df_iot[df_iot["src_ip"] == device_id].reset_index(drop=True)
        n_flows = len(device_df)

        if n_flows < window_size:
            logger.debug(
                f"Device {device_id}: only {n_flows} flows — "
                f"need {window_size} for one window, skipping"
            )
            continue

        n_windows = math.floor((n_flows - window_size) / stride) + 1
        device_records = []

        for w_idx in range(n_windows):
            start = w_idx * stride
            end = start + window_size
            window_df = device_df.iloc[start:end]

            # Label: ATTACK if ANY row in the window is not normal
            # This is the strict definition — a compromised window is any
            # window that contains even one attack flow.
            types_in_window = window_df["type"].unique()
            is_attack = any(t != "normal" for t in types_in_window)
            label = LABEL_ATTACK if is_attack else LABEL_NORMAL

            try:
                features = extract_features(window_df)
            except Exception as exc:
                logger.warning(
                    f"Device {device_id} window {w_idx} (rows {start}-{end}): "
                    f"feature extraction failed — {exc}. Skipping."
                )
                continue

            device_records.append(WindowRecord(
                device_id=device_id,
                features=features,
                label=label,
                window_idx=w_idx,
                flow_start=start,
            ))

        all_records.extend(device_records)
        n_normal = sum(1 for r in device_records if r.label == LABEL_NORMAL)
        n_attack = sum(1 for r in device_records if r.label == LABEL_ATTACK)
        logger.info(
            f"  {device_id}: {len(device_records)} windows "
            f"({n_normal} normal, {n_attack} attack)"
        )

    logger.info(
        f"Total windows: {len(all_records):,} "
        f"({sum(1 for r in all_records if r.label == LABEL_NORMAL):,} normal, "
        f"{sum(1 for r in all_records if r.label == LABEL_ATTACK):,} attack)"
    )
    return WindowDataset(all_records)


# ---------------------------------------------------------------------------
# WindowDataset — container + split logic
# ---------------------------------------------------------------------------

class WindowDataset:
    """Container for all WindowRecord objects with three-way chronological split.

    Normal windows per device are split chronologically: first 80% → train_normal,
    next 14% → enroll_normal, last 6% → test_normal. Chronological ordering
    avoids near-duplicate leakage from the 50%-stride window overlap.
    """

    def __init__(self, records: list[WindowRecord], seed: int = 42) -> None:
        self.records = records
        self._seed = seed
        self.device_ids: list[str] = sorted({r.device_id for r in records})
        self.device_to_idx: dict[str, int] = {
            ip: i for i, ip in enumerate(self.device_ids)
        }

        # Partition into normal / attack
        normal_records = [r for r in records if r.label == LABEL_NORMAL]
        self.attack_records = [r for r in records if r.label == LABEL_ATTACK]

        # Three-way chronological split per device (80 / 14 / 6 %)
        self.train_normal:  list[WindowRecord] = []
        self.enroll_normal: list[WindowRecord] = []
        self.test_normal:   list[WindowRecord] = []

        for device_id in self.device_ids:
            dev_normal = [r for r in normal_records if r.device_id == device_id]
            if not dev_normal:
                continue
            # Sort chronologically by window_idx (earliest traffic first)
            dev_sorted = sorted(dev_normal, key=lambda r: r.window_idx)
            n = len(dev_sorted)

            # Primary split: first 80% → training
            i_train = math.floor(n * TRAIN_SPLIT)
            train_w = dev_sorted[:i_train]

            # Secondary split of remaining 20%:
            # first ENROLL_FRAC (70%) → enroll_normal
            # rest  → test_normal
            held_out = dev_sorted[i_train:]
            i_enroll = math.floor(len(held_out) * ENROLL_FRAC)
            enroll_w = held_out[:i_enroll]
            test_w   = held_out[i_enroll:]

            self.train_normal.extend(train_w)
            self.enroll_normal.extend(enroll_w)
            self.test_normal.extend(test_w)

        logger.info(
            f"WindowDataset: {len(self.device_ids)} devices | "
            f"train_normal={len(self.train_normal)}, "
            f"enroll_normal={len(self.enroll_normal)}, "
            f"test_normal={len(self.test_normal)}, "
            f"attack={len(self.attack_records)}"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a statistics dictionary for logging and reporting."""
        normal_total = len(self.train_normal) + len(self.enroll_normal) + len(self.test_normal)
        per_device = {}
        for device_id in self.device_ids:
            train_n  = sum(1 for r in self.train_normal  if r.device_id == device_id)
            enroll_n = sum(1 for r in self.enroll_normal if r.device_id == device_id)
            test_n   = sum(1 for r in self.test_normal   if r.device_id == device_id)
            atk_n    = sum(1 for r in self.attack_records if r.device_id == device_id)
            per_device[device_id] = {
                "train_normal":  train_n,
                "enroll_normal": enroll_n,
                "test_normal":   test_n,
                "attack":        atk_n,
            }
        return {
            "total_windows":    len(self.records),
            "normal_windows":   normal_total,
            "attack_windows":   len(self.attack_records),
            "train_normal":     len(self.train_normal),
            "enroll_normal":    len(self.enroll_normal),
            "test_normal":      len(self.test_normal),
            "num_devices":      len(self.device_ids),
            "per_device":       per_device,
        }

    def possible_pairs(self, split: str = "train") -> dict[str, int]:
        """Compute exact pair counts for a given split.

        Args:
            split: 'train', 'enroll', or 'test'

        Returns:
            Dict with keys 'positive', 'negative', 'total', 'balanced_total'.
        """
        if split == "train":
            pool = self.train_normal
        elif split == "enroll":
            pool = self.enroll_normal
        else:
            pool = self.test_normal
        per_device: dict[str, list] = {}
        for r in pool:
            per_device.setdefault(r.device_id, []).append(r)

        pos = sum(len(v) * (len(v) - 1) // 2 for v in per_device.values())
        neg = 0
        ids = list(per_device.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                neg += len(per_device[ids[i]]) * len(per_device[ids[j]])

        return {
            "positive":       pos,
            "negative":       neg,
            "total":          pos + neg,
            "balanced_total": min(pos, neg) * 2,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the WindowDataset to disk for fast reloading."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"WindowDataset saved to {path}")

    @staticmethod
    def load(path: Path) -> "WindowDataset":
        """Load a previously saved WindowDataset from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"WindowDataset loaded from {path} ({len(obj.records):,} records)")
        return obj


# ---------------------------------------------------------------------------
# PairDataset — PyTorch Dataset for Siamese training
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """PyTorch Dataset yielding (anchor, pair, label) tuples for contrastive training.

    label=0.0 means same device (positive), label=1.0 means different devices (negative).
    Pairs are pre-generated at construction for reproducibility.

    Args:
        window_dataset: A WindowDataset instance.
        split:          'train', 'enroll', or 'test'.
        n_pairs:        Total pairs to sample (default: balanced min(pos, neg) * 2).
        device_cap:     Max windows per device to cap high-volume devices.
        seed:           Random seed.
    """

    def __init__(
        self,
        window_dataset: WindowDataset,
        split: str = "train",
        n_pairs: Optional[int] = None,
        device_cap: int = DEVICE_SAMPLE_CAP,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.feature_dim = FEATURE_DIM

        if split == "train":
            pool = window_dataset.train_normal
        elif split == "enroll":
            pool = window_dataset.enroll_normal
        else:
            pool = window_dataset.test_normal

        # Build per-device pool with cap
        per_device: dict[str, list[WindowRecord]] = {}
        rng = random.Random(seed)
        for r in pool:
            per_device.setdefault(r.device_id, []).append(r)
        for dev_id in per_device:
            records = per_device[dev_id]
            if len(records) > device_cap:
                per_device[dev_id] = rng.sample(records, device_cap)

        self._per_device = per_device
        self._device_ids = list(per_device.keys())

        # Pre-compute all pairs
        positive_pairs: list[tuple[WindowRecord, WindowRecord]] = []
        negative_pairs: list[tuple[WindowRecord, WindowRecord]] = []

        # Positive: all C(n,2) combos within each device (then sample if too many)
        for records in per_device.values():
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    positive_pairs.append((records[i], records[j]))

        # Negative: for every device pair (i, j), generate cross-device pairs
        for i in range(len(self._device_ids)):
            for j in range(i + 1, len(self._device_ids)):
                recs_i = per_device[self._device_ids[i]]
                recs_j = per_device[self._device_ids[j]]
                for ri in recs_i:
                    for rj in recs_j:
                        negative_pairs.append((ri, rj))

        # Determine target counts
        max_balanced = min(len(positive_pairs), len(negative_pairs))
        if n_pairs is None:
            target_each = max_balanced
        else:
            target_each = n_pairs // 2

        target_each = min(target_each, max_balanced)

        # Sample to target
        rng.shuffle(positive_pairs)
        rng.shuffle(negative_pairs)
        selected_pos = positive_pairs[:target_each]
        selected_neg = negative_pairs[:target_each]

        # Interleave for stable training dynamics (alternating pos/neg)
        self._pairs: list[tuple[WindowRecord, WindowRecord, int]] = []
        for p, n in zip(selected_pos, selected_neg):
            self._pairs.append((p[0], p[1], LABEL_NORMAL))   # label=0: same device
            self._pairs.append((n[0], n[1], LABEL_ATTACK))   # label=1: diff device

        rng.shuffle(self._pairs)

        logger.info(
            f"PairDataset ({split}): {len(selected_pos):,} positive + "
            f"{len(selected_neg):,} negative = {len(self._pairs):,} total pairs | "
            f"device_cap={device_cap}, devices={len(self._device_ids)}"
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record_a, record_b, label = self._pairs[idx]
        anchor = torch.from_numpy(record_a.features)   # already float32
        pair   = torch.from_numpy(record_b.features)   # already float32
        lbl    = torch.tensor(float(label), dtype=torch.float32)
        return anchor, pair, lbl

    @property
    def n_devices(self) -> int:
        return len(self._device_ids)

    @property
    def device_ids(self) -> list[str]:
        return self._device_ids.copy()
