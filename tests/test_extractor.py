"""
Tests for src/features/extractor.py

Run with:  pytest tests/test_extractor.py -v

Test philosophy:
- Use REAL data slices from train_test_network.csv wherever possible
- Synthetic DataFrames only for edge-case / unit tests
- Every test has a docstring explaining *why* it matters scientifically
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.features.extractor import (
    FEATURE_DIM,
    FEATURE_NAMES,
    WINDOW_SIZE,
    extract_features,
    _shannon_entropy,
    _ratio,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TON_IOT_PATH = Path("data/raw/ton_iot/train_test_network.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load the full TON_IoT network CSV once per test session."""
    if not TON_IOT_PATH.exists():
        pytest.skip("TON_IoT dataset not found — skipping real-data tests")
    return pd.read_csv(TON_IOT_PATH, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def normal_df(real_df: pd.DataFrame) -> pd.DataFrame:
    """Normal-traffic rows only."""
    return real_df[real_df["type"] == "normal"].reset_index(drop=True)


@pytest.fixture
def synthetic_window() -> pd.DataFrame:
    """Minimal synthetic window with all expected columns."""
    n = WINDOW_SIZE
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "src_ip":               ["192.168.1.103"] * n,
        "src_port":             rng.integers(1024, 65535, n).astype(str),
        "dst_ip":               ["8.8.8.8"] * (n // 2) + ["1.1.1.1"] * (n // 2),
        "dst_port":             (["53"] * (n // 2) + ["443"] * (n // 2)),
        "proto":                (["udp"] * (n // 2) + ["tcp"] * (n // 2)),
        "service":              (["dns"] * (n // 2) + ["ssl"] * (n // 2)),
        "duration":             rng.uniform(0.0, 1.0, n).astype(str),
        "src_bytes":            rng.integers(0, 500, n).astype(str),
        "dst_bytes":            rng.integers(0, 500, n).astype(str),
        "conn_state":           ["SF"] * (n // 2) + ["S0"] * (n // 2),
        "missed_bytes":         ["0"] * n,
        "src_pkts":             rng.integers(1, 5, n).astype(str),
        "src_ip_bytes":         rng.integers(50, 600, n).astype(str),
        "dst_pkts":             rng.integers(1, 5, n).astype(str),
        "dst_ip_bytes":         rng.integers(50, 600, n).astype(str),
        "dns_query":            ["example.com"] * (n // 2) + ["-"] * (n // 2),
        "dns_qclass":           ["-"] * n,
        "dns_qtype":            ["-"] * n,
        "dns_rcode":            ["0"] * (n // 2) + ["-"] * (n // 2),
        "dns_AA":               ["-"] * n,
        "dns_RD":               ["-"] * n,
        "dns_RA":               ["-"] * n,
        "dns_rejected":         ["-"] * n,
        "ssl_version":          ["-"] * (n // 2) + ["TLSv12"] * (n // 2),
        "ssl_cipher":           ["-"] * n,
        "ssl_resumed":          ["-"] * (n // 2) + ["F"] * (n // 2),
        "ssl_established":      ["-"] * n,
        "ssl_subject":          ["-"] * n,
        "ssl_issuer":           ["-"] * n,
        "http_trans_depth":     ["-"] * n,
        "http_method":          ["-"] * n,
        "http_uri":             ["-"] * n,
        "http_version":         ["-"] * n,
        "http_request_body_len": ["0"] * n,
        "http_response_body_len": ["0"] * n,
        "http_status_code":     ["-"] * n,
        "http_user_agent":      ["-"] * n,
        "http_orig_mime_types": ["-"] * n,
        "http_resp_mime_types": ["-"] * n,
        "weird_name":           ["-"] * n,
        "weird_addl":           ["-"] * n,
        "weird_notice":         ["-"] * n,
        "label":                ["0"] * n,
        "type":                 ["normal"] * n,
    })


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_ratio_normal(self):
        """_ratio should return exact quotient for non-zero denominator."""
        assert _ratio(3.0, 4.0) == pytest.approx(0.75)

    def test_ratio_zero_denominator(self):
        """_ratio must return 0.0 when denominator is zero — no ZeroDivisionError."""
        assert _ratio(5.0, 0.0) == 0.0

    def test_entropy_uniform(self):
        """Uniform distribution over k categories has entropy log2(k)."""
        vals = ["a", "b", "c", "d"]
        assert _shannon_entropy(vals) == pytest.approx(2.0)

    def test_entropy_single_value(self):
        """Single repeated value has zero entropy."""
        assert _shannon_entropy(["x", "x", "x"]) == pytest.approx(0.0)

    def test_entropy_empty(self):
        """Empty input returns 0.0 — should never trigger NaN downstream."""
        assert _shannon_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_shape_synthetic(self, synthetic_window):
        """Output must always be exactly (60,) — the Siamese encoder input size."""
        vec = extract_features(synthetic_window)
        assert vec.shape == (FEATURE_DIM,), f"Got shape {vec.shape}"

    def test_dtype_float32(self, synthetic_window):
        """Must be float32 — not float64. PyTorch default tensor type is float32."""
        vec = extract_features(synthetic_window)
        assert vec.dtype == np.float32

    def test_no_nan_or_inf(self, synthetic_window):
        """No NaN or Inf values — would corrupt the scaler and training loss."""
        vec = extract_features(synthetic_window)
        assert np.all(np.isfinite(vec)), f"Non-finite values at indices: {np.where(~np.isfinite(vec))[0]}"

    def test_feature_names_length(self):
        """FEATURE_NAMES registry must have exactly FEATURE_DIM entries."""
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_empty_dataframe_raises(self):
        """Empty DataFrame must raise ValueError — not silently return zeros."""
        with pytest.raises(ValueError, match="empty"):
            extract_features(pd.DataFrame())


# ---------------------------------------------------------------------------
# Semantic / correctness tests
# ---------------------------------------------------------------------------

class TestSemantics:
    def test_flow_count_is_n(self, synthetic_window):
        """Feature index 0 (flow_count) must equal the actual window size."""
        vec = extract_features(synthetic_window)
        assert vec[0] == pytest.approx(float(WINDOW_SIZE))

    def test_all_tcp_sets_tcp_ratio_one(self, synthetic_window):
        """If all flows are TCP, tcp_ratio (idx 13) must be 1.0."""
        df = synthetic_window.copy()
        df["proto"] = "tcp"
        vec = extract_features(df)
        assert vec[13] == pytest.approx(1.0)   # tcp_ratio
        assert vec[14] == pytest.approx(0.0)   # udp_ratio

    def test_all_s0_sets_scan_indicator(self, synthetic_window):
        """All S0 connections → s0_ratio (idx 21) = 1.0. This is a scan pattern."""
        df = synthetic_window.copy()
        df["conn_state"] = "S0"
        vec = extract_features(df)
        assert vec[21] == pytest.approx(1.0)   # s0_ratio
        assert vec[22] == pytest.approx(0.0)   # sf_ratio

    def test_bytes_out_in_ratio_symmetric(self, synthetic_window):
        """Equal src and dst bytes → bytes_out_in_ratio (idx 57) = 1.0."""
        df = synthetic_window.copy()
        df["src_bytes"] = "100"
        df["dst_bytes"] = "100"
        vec = extract_features(df)
        assert vec[57] == pytest.approx(1.0)

    def test_missing_sentinels_handled(self, synthetic_window):
        """'-' values throughout must not cause NaN or crash."""
        df = synthetic_window.copy()
        df["duration"] = "-"
        df["src_bytes"] = "-"
        vec = extract_features(df)
        assert np.all(np.isfinite(vec))

    def test_dns_entropy_nonzero_for_varied_queries(self, synthetic_window):
        """Varied DNS queries → entropy (idx 38) > 0. DGA detection depends on this."""
        df = synthetic_window.copy()
        df["dns_query"] = [f"host{i}.evil.com" for i in range(len(df))]
        vec = extract_features(df)
        assert vec[38] > 0.0  # dns_query_entropy

    def test_nxdomain_ratio_correct(self, synthetic_window):
        """Half NXDOMAIN (rcode=3) → nxdomain_ratio (idx 37) = 0.5."""
        df = synthetic_window.copy()
        n = len(df)
        df["dns_rcode"] = ["3"] * (n // 2) + ["0"] * (n // 2)
        vec = extract_features(df)
        assert vec[37] == pytest.approx(0.5)

    def test_fan_out_ratio_high_for_scanner(self, synthetic_window):
        """Device contacting many unique IPs → high fan_out_ratio (idx 59)."""
        df = synthetic_window.copy()
        df["dst_ip"] = [f"10.0.0.{i}" for i in range(len(df))]
        vec = extract_features(df)
        assert vec[59] > 0.5  # fan_out_ratio


# ---------------------------------------------------------------------------
# Real-data smoke tests
# ---------------------------------------------------------------------------

class TestRealData:
    def test_shape_on_real_normal_slice(self, normal_df):
        """First WINDOW_SIZE normal rows must produce correct shape — real data test."""
        window = normal_df.iloc[:WINDOW_SIZE]
        vec = extract_features(window)
        assert vec.shape == (FEATURE_DIM,)

    def test_no_nan_on_real_data(self, normal_df):
        """No NaN on a real normal window — validates missing-value handling."""
        window = normal_df.iloc[:WINDOW_SIZE]
        vec = extract_features(window)
        assert np.all(np.isfinite(vec)), (
            f"Non-finite at indices: {np.where(~np.isfinite(vec))[0]}"
        )

    def test_all_device_ips_produce_valid_vectors(self, normal_df):
        """Each local IoT device IP must produce a valid feature vector."""
        device_ips = [ip for ip in normal_df["src_ip"].unique()
                      if str(ip).startswith("192.168.1.")]
        for ip in device_ips:
            device_flows = normal_df[normal_df["src_ip"] == ip]
            if len(device_flows) < WINDOW_SIZE:
                continue  # not enough flows for this device — skip
            window = device_flows.iloc[:WINDOW_SIZE]
            vec = extract_features(window)
            assert vec.shape == (FEATURE_DIM,), f"Bad shape for device {ip}"
            assert np.all(np.isfinite(vec)), f"NaN/Inf for device {ip}"

    def test_normal_vs_attack_vectors_differ(self, real_df):
        """Normal and attack windows for same device should produce different vectors.

        This is a sanity check that the features are sensitive to attack behavior.
        If they're identical, the Siamese network cannot learn to distinguish them.
        """
        ip = "192.168.1.193"  # IP with both normal and backdoor/ransomware traffic
        normal_window = real_df[
            (real_df["src_ip"] == ip) & (real_df["type"] == "normal")
        ].iloc[:WINDOW_SIZE]
        attack_window = real_df[
            (real_df["src_ip"] == ip) & (real_df["type"] != "normal")
        ].iloc[:WINDOW_SIZE]

        if len(normal_window) < WINDOW_SIZE or len(attack_window) < WINDOW_SIZE:
            pytest.skip(f"Not enough flows for device {ip}")

        vec_normal = extract_features(normal_window)
        vec_attack  = extract_features(attack_window)
        l2_distance = float(np.linalg.norm(vec_normal - vec_attack))
        assert l2_distance > 0.0, (
            "Normal and attack feature vectors are identical — "
            "features have zero discriminative power"
        )
