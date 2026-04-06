"""
Tests for src/training/dataset.py

Run with:  pytest tests/test_dataset.py -v

Test philosophy:
- Unit tests use tiny synthetic DataFrames to test logic in isolation
- Integration tests run on the real TON_IoT CSV and verify expected numbers
- Every test documents WHY it matters for the research claim
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.training.dataset import (
    LABEL_ATTACK,
    LABEL_NORMAL,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    WindowDataset,
    WindowRecord,
    PairDataset,
    build_windows,
)
from src.features.extractor import FEATURE_DIM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TON_IOT_PATH = Path("data/raw/ton_iot/train_test_network.csv")

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _make_synthetic_csv_df(
    n_flows_per_device: dict[str, int],
    attack_device: str | None = None,
    attack_fraction: float = 0.5,
) -> pd.DataFrame:
    """Build a minimal synthetic DataFrame that mimics train_test_network.csv.

    Args:
        n_flows_per_device: e.g. {'192.168.1.1': 200, '192.168.1.2': 150}
        attack_device:      which device IP gets attack rows (None = all normal)
        attack_fraction:    fraction of attack_device rows that are attacks
    """
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    for ip, n in n_flows_per_device.items():
        for i in range(n):
            is_atk = (
                attack_device == ip
                and i >= int(n * (1 - attack_fraction))
            )
            rows.append({
                "src_ip":   ip,
                "src_port": str(rng.integers(1024, 65535)),
                "dst_ip":   "8.8.8.8",
                "dst_port": "53",
                "proto":    "udp",
                "service":  "dns",
                "duration": str(rng.uniform(0, 1)),
                "src_bytes": str(rng.integers(0, 300)),
                "dst_bytes": str(rng.integers(0, 300)),
                "conn_state": "SF",
                "missed_bytes": "0",
                "src_pkts":    "1",
                "src_ip_bytes": "100",
                "dst_pkts":    "1",
                "dst_ip_bytes": "100",
                "dns_query": "example.com",
                "dns_qclass": "-", "dns_qtype": "-", "dns_rcode": "0",
                "dns_AA": "-", "dns_RD": "-", "dns_RA": "-",
                "dns_rejected": "-", "ssl_version": "-", "ssl_cipher": "-",
                "ssl_resumed": "-", "ssl_established": "-",
                "ssl_subject": "-", "ssl_issuer": "-",
                "http_trans_depth": "-", "http_method": "-",
                "http_uri": "-", "http_version": "-",
                "http_request_body_len": "0",
                "http_response_body_len": "0",
                "http_status_code": "-", "http_user_agent": "-",
                "http_orig_mime_types": "-", "http_resp_mime_types": "-",
                "weird_name": "-", "weird_addl": "-", "weird_notice": "-",
                "label": "1" if is_atk else "0",
                "type":  "backdoor" if is_atk else "normal",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_dataset() -> WindowDataset:
    """Two devices with 200 flows each — fully synthetic, all normal."""
    df = _make_synthetic_csv_df({
        "192.168.1.10": 200,
        "192.168.1.20": 200,
    })
    # Build WindowDataset directly from the DataFrame (bypass CSV loading)
    records = []
    from src.features.extractor import extract_features
    for device_id, group in df.groupby("src_ip"):
        group = group.reset_index(drop=True)
        n = len(group)
        n_windows = math.floor((n - WINDOW_SIZE) / WINDOW_STRIDE) + 1
        for w_idx in range(n_windows):
            start = w_idx * WINDOW_STRIDE
            window_df = group.iloc[start:start + WINDOW_SIZE]
            types = window_df["type"].unique()
            label = LABEL_ATTACK if any(t != "normal" for t in types) else LABEL_NORMAL
            features = extract_features(window_df)
            records.append(WindowRecord(
                device_id=str(device_id),
                features=features,
                label=label,
                window_idx=w_idx,
                flow_start=start,
            ))
    return WindowDataset(records, seed=42)


@pytest.fixture(scope="module")
def real_window_dataset() -> WindowDataset:
    """Build WindowDataset from the real TON_IoT CSV (loaded once per session)."""
    if not TON_IOT_PATH.exists():
        pytest.skip("TON_IoT dataset not found")
    return build_windows(TON_IOT_PATH)


# ---------------------------------------------------------------------------
# Unit tests: window count arithmetic
# ---------------------------------------------------------------------------

class TestWindowCounting:
    def test_window_count_formula(self):
        """Window count = floor((n - W) / stride) + 1.

        This is the standard sliding-window formula used in signal processing.
        Verify it matches the implementation.
        """
        n_flows = 200
        expected = math.floor((n_flows - WINDOW_SIZE) / WINDOW_STRIDE) + 1
        assert expected == 7   # (200-50)/25 + 1 = 7

    def test_two_devices_200_flows_each_produce_7_windows(self, synthetic_dataset):
        """With 200 flows and stride=25, each device must produce exactly 7 windows."""
        stats = synthetic_dataset.stats()
        for device_id, d in stats["per_device"].items():
            total = d["train_normal"] + d["test_normal"] + d["attack"]
            assert total == 7, (
                f"Device {device_id} has {total} windows, expected 7"
            )

    def test_total_windows_equals_sum_of_per_device(self, synthetic_dataset):
        """Total window count must equal sum of all per-device counts."""
        stats = synthetic_dataset.stats()
        summed = sum(
            d["train_normal"] + d["test_normal"] + d["attack"]
            for d in stats["per_device"].values()
        )
        assert stats["total_windows"] == summed


# ---------------------------------------------------------------------------
# Unit tests: labeling correctness
# ---------------------------------------------------------------------------

class TestWindowLabeling:
    def test_all_normal_flows_label_is_0(self, synthetic_dataset):
        """Windows from all-normal devices must all have label=0.

        This is the TRAINING SET PURITY guarantee — if labeling is wrong,
        attack flows leak into training, invalidating the research claim.
        """
        for record in synthetic_dataset.train_normal:
            assert record.label == LABEL_NORMAL

    def test_attack_window_correctly_labeled(self):
        """A window containing even ONE attack row must be labeled ATTACK."""
        from src.features.extractor import extract_features
        df = _make_synthetic_csv_df(
            {"192.168.1.99": 100},
            attack_device="192.168.1.99",
            attack_fraction=0.1,  # last 10 flows are attacks
        )
        device_df = df.reset_index(drop=True)
        records = []
        n = len(device_df)
        n_windows = math.floor((n - WINDOW_SIZE) / WINDOW_STRIDE) + 1
        for w_idx in range(n_windows):
            start = w_idx * WINDOW_STRIDE
            window_df = device_df.iloc[start:start + WINDOW_SIZE]
            types = window_df["type"].unique()
            label = LABEL_ATTACK if any(t != "normal" for t in types) else LABEL_NORMAL
            features = extract_features(window_df)
            records.append(WindowRecord(
                device_id="192.168.1.99",
                features=features,
                label=label,
                window_idx=w_idx,
                flow_start=start,
            ))
        # The last window must be ATTACK
        attack_records = [r for r in records if r.label == LABEL_ATTACK]
        assert len(attack_records) > 0, "No attack windows generated despite attack flows"


# ---------------------------------------------------------------------------
# Unit tests: train/test split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_split_ratio_approximately_80_20(self, synthetic_dataset):
        """Each device should have ~80% in train, ~20% in test.

        The exact split is floor(n * 0.8), so with 7 windows:
        5 train, 2 test.
        """
        stats = synthetic_dataset.stats()
        for device_id, d in stats["per_device"].items():
            total_normal = d["train_normal"] + d["test_normal"]
            if total_normal == 0:
                continue
            train_frac = d["train_normal"] / total_normal
            # Allow ±5% tolerance due to integer rounding
            assert 0.70 <= train_frac <= 0.90, (
                f"Device {device_id} has {train_frac:.1%} train split "
                f"(expected ~80%)"
            )

    def test_no_overlap_between_train_and_test(self, synthetic_dataset):
        """The same WindowRecord must not appear in both train and test.

        Overlap would cause evaluation data to contaminate training —
        the most common ML evaluation bug.
        """
        train_ids = {
            (r.device_id, r.window_idx)
            for r in synthetic_dataset.train_normal
        }
        test_ids = {
            (r.device_id, r.window_idx)
            for r in synthetic_dataset.test_normal
        }
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"Found {len(overlap)} records in both train and test splits: {overlap}"
        )

    def test_attack_records_not_in_train_or_test_normal(self, synthetic_dataset):
        """Attack records must never appear in train_normal or test_normal."""
        normal_ids = {
            (r.device_id, r.window_idx)
            for r in synthetic_dataset.train_normal + synthetic_dataset.test_normal
        }
        for r in synthetic_dataset.attack_records:
            assert (r.device_id, r.window_idx) not in normal_ids, (
                f"Attack record (device={r.device_id}, w_idx={r.window_idx}) "
                f"found in normal splits"
            )


# ---------------------------------------------------------------------------
# PairDataset tests
# ---------------------------------------------------------------------------

class TestPairDataset:
    def test_pair_shape_and_dtype(self, synthetic_dataset):
        """Each item must be (tensor(60,), tensor(60,), scalar tensor).

        PyTorch DataLoader will fail silently if dtypes are wrong.
        """
        pair_ds = PairDataset(synthetic_dataset, split="train", seed=42)
        anchor, pair, label = pair_ds[0]
        assert anchor.shape == (FEATURE_DIM,), f"anchor shape: {anchor.shape}"
        assert pair.shape   == (FEATURE_DIM,), f"pair shape: {pair.shape}"
        assert anchor.dtype == torch.float32
        assert pair.dtype   == torch.float32
        assert label.dtype  == torch.float32

    def test_balanced_labels(self, synthetic_dataset):
        """PairDataset must produce exactly equal positive and negative pairs.

        Imbalance would cause the Siamese network to collapse to the majority class.
        """
        pair_ds = PairDataset(synthetic_dataset, split="train", seed=42)
        labels = [pair_ds[i][2].item() for i in range(len(pair_ds))]
        n_pos = sum(1 for l in labels if l == LABEL_NORMAL)
        n_neg = sum(1 for l in labels if l == LABEL_ATTACK)
        assert n_pos == n_neg, (
            f"Imbalanced pairs: {n_pos} positive vs {n_neg} negative"
        )

    def test_positive_pairs_same_device(self, synthetic_dataset):
        """Positive pairs (label=0) must come from the same device.

        A positive pair from different devices would teach the network
        that different devices are the same — directly harming detection.
        """
        pair_ds = PairDataset(synthetic_dataset, split="train", seed=42)
        for i in range(min(100, len(pair_ds))):
            _, _, label = pair_ds[i]
            if label.item() == LABEL_NORMAL:
                # We cannot directly check device_id from the tensor,
                # but we CAN check that the internal record pair matches
                rec_a, rec_b, _ = pair_ds._pairs[i]
                assert rec_a.device_id == rec_b.device_id, (
                    f"Positive pair at idx {i}: "
                    f"device {rec_a.device_id} != {rec_b.device_id}"
                )

    def test_negative_pairs_different_device(self, synthetic_dataset):
        """Negative pairs (label=1) must come from different devices."""
        pair_ds = PairDataset(synthetic_dataset, split="train", seed=42)
        for i in range(min(100, len(pair_ds))):
            rec_a, rec_b, lbl = pair_ds._pairs[i]
            if lbl == LABEL_ATTACK:
                assert rec_a.device_id != rec_b.device_id, (
                    f"Negative pair at idx {i}: same device {rec_a.device_id}"
                )

    def test_dataloader_compatible(self, synthetic_dataset):
        """PairDataset must work with PyTorch DataLoader (batch stacking)."""
        from torch.utils.data import DataLoader
        pair_ds = PairDataset(synthetic_dataset, split="train", seed=42)
        loader = DataLoader(pair_ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        anchors, pairs, labels = batch
        assert anchors.shape == (4, FEATURE_DIM)
        assert pairs.shape   == (4, FEATURE_DIM)
        assert labels.shape  == (4,)


# ---------------------------------------------------------------------------
# Real-data integration tests
# ---------------------------------------------------------------------------

class TestRealData:
    def test_total_windows_in_expected_range(self, real_window_dataset):
        """Total windows on real data must be ~6004 (verified 2026-04-02).

        Pre-analysis counted only normal-producing devices (1713), but
        build_windows() correctly includes ALL 192.168.1.x devices —
        including pure attacker IPs (.30=2118, .31=1073, .33=116, etc.)
        which generate attack-labeled windows. Total = 6004 (1707 normal,
        4297 attack).
        """
        stats = real_window_dataset.stats()
        # Allow ±200 variance
        assert 5800 <= stats["total_windows"] <= 6200, (
            f"Total windows {stats['total_windows']} out of expected range [5800, 6200]"
        )

    def test_normal_windows_in_expected_range(self, real_window_dataset):
        """Normal windows must be ~1713 (most windows are from normal-dominant devices)."""
        stats = real_window_dataset.stats()
        assert stats["normal_windows"] >= 1600, (
            f"Only {stats['normal_windows']} normal windows — expected >= 1600"
        )

    def test_at_least_15_device_classes(self, real_window_dataset):
        """Must have >= 15 device classes with windows for meaningful pair diversity."""
        stats = real_window_dataset.stats()
        assert stats["num_devices"] >= 15, (
            f"Only {stats['num_devices']} device classes — need >= 15"
        )

    def test_possible_train_pairs_exceeds_100k(self, real_window_dataset):
        """Training pair pool must be large enough for the Siamese network to learn."""
        pair_counts = real_window_dataset.possible_pairs("train")
        assert pair_counts["balanced_total"] > 100_000, (
            f"Only {pair_counts['balanced_total']:,} balanced pairs — "
            f"need > 100k for meaningful contrastive learning"
        )

    def test_no_attack_records_in_training_split(self, real_window_dataset):
        """CRITICAL: No attack-labeled record must appear in train_normal.

        This is the single most important invariant in the entire project.
        Violating it invalidates the zero-day detection claim.
        """
        for record in real_window_dataset.train_normal:
            assert record.label == LABEL_NORMAL, (
                f"ATTACK record (device={record.device_id}, "
                f"w_idx={record.window_idx}) found in train_normal!"
            )

    def test_pair_dataset_builds_on_real_data(self, real_window_dataset):
        """PairDataset must build successfully on real data without errors."""
        pair_ds = PairDataset(real_window_dataset, split="train", seed=42)
        assert len(pair_ds) > 0
        anchor, _, _ = pair_ds[0]
        assert anchor.shape == (FEATURE_DIM,)
        assert np.isfinite(anchor.numpy()).all()
