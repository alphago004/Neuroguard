"""
Feature extractor for NEUROGUARD behavioral fingerprinting.

Converts a 50-flow window (DataFrame of Zeek connection-log rows) into a
60-dimensional float32 feature vector representing the behavioral signature
of one IoT device during that window.

Design notes
------------
- Input data: TON_IoT train_test_network.csv (Zeek format, BOM-encoded)
- Window unit: 50 consecutive flows from a single src_ip
- All string sentinel values ('-', '', '(empty)') treated as missing
- Division-by-zero guards return 0.0 (not NaN) so the scaler never sees NaN
- Shannon entropy uses log2; 0 * log2(0) → 0 by convention (scipy behavior)
- Feature indices are STABLE — do not reorder without updating CLAUDE.md §5

Feature layout (60 total)
--------------------------
Index  0– 4  : Volume / byte counts (5)
Index  5– 8  : Packet stats (4)
Index  9–12  : Duration stats (4)
Index 13–16  : Protocol ratios (4)
Index 17–20  : Service ratios (4)
Index 21–27  : Connection-state ratios (7)
Index 28–30  : Connectivity / fan-out (3)
Index 31–34  : Port distribution (4)
Index 35–38  : DNS features (4)
Index 39–41  : HTTP features (3)
Index 42–43  : SSL features (2)
Index 44–44  : Missed-bytes ratio (1)
Index 45–50  : Byte-size percentiles — src (3) + dst (3)
Index 51–53  : Duration percentiles (3)
Index 54–56  : Inter-flow byte-gap stats (3)
Index 57–59  : Asymmetry / ratio features (3)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 50          # flows per window (design decision 2026-04-02)
FEATURE_DIM: int = 60          # output vector length

# Zeek conn_state values and their security meaning
# REF: https://docs.zeek.org/en/master/scripts/base/protocols/conn/main.zeek.html
_CONN_STATES: tuple[str, ...] = (
    "S0",    # SYN only — likely scan / no response
    "S1",    # established, not closed
    "SF",    # normal full open-close
    "REJ",   # RST in reply to SYN — port closed
    "S2",    # established, close attempt by originator
    "S3",    # established, close attempt by responder
    "RSTO",  # reset by originator
    "RSTS",  # reset by responder
    "RSTOS0",# originator SYN then RST
    "SHR",   # responder SYN-ACK only
    "SH",    # SYN-ACK only — half-open
    "OTH",   # mid-stream traffic / no SYN seen
)

_MISSING: frozenset[str] = frozenset({"", "-", "(empty)"})

# Well-known port threshold (IANA)
_WELL_KNOWN_PORT_MAX: int = 1023
_EPHEMERAL_PORT_MIN: int = 49152


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> Optional[float]:
    """Convert a cell value to float; return None for missing sentinels."""
    if value is None:
        return None
    s = str(value).strip()
    if s in _MISSING:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or 0.0 if denominator == 0."""
    return numerator / denominator if denominator > 0.0 else 0.0


def _shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy (bits) of a categorical distribution.

    Args:
        values: List of string tokens (e.g. DNS domain labels).

    Returns:
        Entropy in bits; 0.0 for empty or uniform-single-value lists.
    """
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = len(values)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def _percentiles(arr: list[float], qs: tuple[float, ...] = (25.0, 50.0, 75.0)) -> list[float]:
    """Return requested percentiles; all-zero list for empty input."""
    if not arr:
        return [0.0] * len(qs)
    a = np.array(arr, dtype=np.float64)
    return [float(np.percentile(a, q)) for q in qs]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(window_df: pd.DataFrame) -> np.ndarray:
    """Extract a 60-dimensional behavioral feature vector from a flow window.

    The window should contain exactly WINDOW_SIZE (50) Zeek connection-log
    rows for a single device (src_ip). Fewer rows are accepted at stream
    edges but will produce noisier vectors — the training pipeline should
    filter them out.

    Args:
        window_df: DataFrame with Zeek columns from train_test_network.csv.
                   Must include at minimum: proto, service, duration,
                   src_bytes, dst_bytes, src_pkts, dst_pkts, conn_state,
                   missed_bytes, src_ip_bytes, dst_ip_bytes, dst_port,
                   src_port, dns_query, dns_rcode, http_request_body_len,
                   http_response_body_len, ssl_version, ssl_resumed.

    Returns:
        numpy array of shape (60,), dtype float32.

    Raises:
        ValueError: If window_df is empty or missing required columns.
    """
    if window_df.empty:
        raise ValueError("window_df is empty — cannot extract features")

    n = len(window_df)  # actual flow count (≤ WINDOW_SIZE at stream edges)

    # ------------------------------------------------------------------
    # Pre-extract numeric series with NaN for missing values
    # ------------------------------------------------------------------
    def _col_floats(col: str) -> list[float]:
        """Return non-None float values for a column."""
        out = []
        if col not in window_df.columns:
            return out
        for v in window_df[col]:
            f = _safe_float(v)
            if f is not None:
                out.append(f)
        return out

    def _col_strings(col: str) -> list[str]:
        """Return non-missing string values for a column."""
        out = []
        if col not in window_df.columns:
            return out
        for v in window_df[col]:
            s = str(v).strip() if v is not None else ""
            if s not in _MISSING:
                out.append(s)
        return out

    src_bytes_list   = _col_floats("src_bytes")
    dst_bytes_list   = _col_floats("dst_bytes")
    src_pkts_list    = _col_floats("src_pkts")
    dst_pkts_list    = _col_floats("dst_pkts")
    duration_list    = _col_floats("duration")
    missed_list      = _col_floats("missed_bytes")
    src_ip_bytes_list = _col_floats("src_ip_bytes")
    dst_ip_bytes_list = _col_floats("dst_ip_bytes")
    dst_port_list    = _col_floats("dst_port")
    src_port_list    = _col_floats("src_port")
    http_req_list    = _col_floats("http_request_body_len")
    http_resp_list   = _col_floats("http_response_body_len")

    proto_list       = _col_strings("proto")
    service_list     = _col_strings("service")
    conn_state_list  = _col_strings("conn_state")
    dns_query_list   = _col_strings("dns_query")
    dns_rcode_list   = _col_strings("dns_rcode")
    ssl_version_list = _col_strings("ssl_version")
    ssl_resumed_list = _col_strings("ssl_resumed")

    total_src_bytes  = sum(src_bytes_list)
    total_dst_bytes  = sum(dst_bytes_list)
    total_src_pkts   = sum(src_pkts_list)
    total_dst_pkts   = sum(dst_pkts_list)
    total_bytes      = total_src_bytes + total_dst_bytes

    # ------------------------------------------------------------------
    # Feature group 0: Volume / byte counts  [idx 0–4]
    # ------------------------------------------------------------------
    f00_flow_count         = float(n)
    f01_total_src_bytes    = total_src_bytes
    f02_total_dst_bytes    = total_dst_bytes
    f03_total_src_ip_bytes = sum(src_ip_bytes_list)
    f04_total_dst_ip_bytes = sum(dst_ip_bytes_list)

    # ------------------------------------------------------------------
    # Feature group 1: Packet stats  [idx 5–8]
    # ------------------------------------------------------------------
    f05_mean_src_pkts = _ratio(total_src_pkts, n)
    f06_std_src_pkts  = float(np.std(src_pkts_list)) if src_pkts_list else 0.0
    f07_mean_dst_pkts = _ratio(total_dst_pkts, n)
    f08_std_dst_pkts  = float(np.std(dst_pkts_list)) if dst_pkts_list else 0.0

    # ------------------------------------------------------------------
    # Feature group 2: Duration stats  [idx 9–12]
    # ------------------------------------------------------------------
    f09_mean_duration = _ratio(sum(duration_list), len(duration_list)) if duration_list else 0.0
    f10_std_duration  = float(np.std(duration_list)) if duration_list else 0.0
    f11_min_duration  = float(min(duration_list)) if duration_list else 0.0
    f12_max_duration  = float(max(duration_list)) if duration_list else 0.0

    # ------------------------------------------------------------------
    # Feature group 3: Protocol ratios  [idx 13–16]
    # ------------------------------------------------------------------
    f13_tcp_ratio   = _ratio(proto_list.count("tcp"), n)
    f14_udp_ratio   = _ratio(proto_list.count("udp"), n)
    f15_icmp_ratio  = _ratio(proto_list.count("icmp"), n)
    _other_proto    = n - proto_list.count("tcp") - proto_list.count("udp") - proto_list.count("icmp")
    f16_other_proto_ratio = _ratio(_other_proto, n)

    # ------------------------------------------------------------------
    # Feature group 4: Service ratios  [idx 17–20]
    # ------------------------------------------------------------------
    f17_dns_service_ratio  = _ratio(service_list.count("dns"), n)
    f18_http_service_ratio = _ratio(service_list.count("http"), n)
    f19_ssl_service_ratio  = _ratio(service_list.count("ssl"), n)
    _known_services = service_list.count("dns") + service_list.count("http") + service_list.count("ssl")
    f20_other_service_ratio = _ratio(n - _known_services, n)

    # ------------------------------------------------------------------
    # Feature group 5: Connection-state ratios  [idx 21–27]
    # ------------------------------------------------------------------
    # S0 = scan indicator (SYN only, no response) — high diagnostic value
    f21_s0_ratio   = _ratio(conn_state_list.count("S0"), n)
    f22_sf_ratio   = _ratio(conn_state_list.count("SF"), n)
    f23_rej_ratio  = _ratio(conn_state_list.count("REJ"), n)
    f24_oth_ratio  = _ratio(conn_state_list.count("OTH"), n)
    f25_rsto_ratio = _ratio(conn_state_list.count("RSTO"), n)
    f26_shr_ratio  = _ratio(conn_state_list.count("SHR"), n)
    # Catch all remaining states (S1, S2, S3, RSTS, RSTOS0, SH, etc.)
    _counted = (conn_state_list.count("S0") + conn_state_list.count("SF") +
                conn_state_list.count("REJ") + conn_state_list.count("OTH") +
                conn_state_list.count("RSTO") + conn_state_list.count("SHR"))
    f27_other_state_ratio = _ratio(n - _counted, n)

    # ------------------------------------------------------------------
    # Feature group 6: Connectivity / fan-out  [idx 28–30]
    # ------------------------------------------------------------------
    unique_dst_ips   = window_df["dst_ip"].nunique() if "dst_ip" in window_df.columns else 0
    unique_dst_ports = len(set(int(p) for p in dst_port_list if 0 <= p <= 65535))
    unique_src_ports = len(set(int(p) for p in src_port_list if 0 <= p <= 65535))

    f28_unique_dst_ips   = float(unique_dst_ips)
    f29_unique_dst_ports = float(unique_dst_ports)
    f30_unique_src_ports = float(unique_src_ports)

    # ------------------------------------------------------------------
    # Feature group 7: Port distribution  [idx 31–34]
    # ------------------------------------------------------------------
    valid_dst_ports  = [p for p in dst_port_list if 0 <= p <= 65535]
    f31_mean_dst_port       = float(np.mean(valid_dst_ports)) if valid_dst_ports else 0.0
    f32_std_dst_port        = float(np.std(valid_dst_ports)) if valid_dst_ports else 0.0
    f33_well_known_port_ratio = _ratio(
        sum(1 for p in valid_dst_ports if p <= _WELL_KNOWN_PORT_MAX),
        len(valid_dst_ports)
    )
    f34_ephemeral_port_ratio = _ratio(
        sum(1 for p in valid_dst_ports if p >= _EPHEMERAL_PORT_MIN),
        len(valid_dst_ports)
    )

    # ------------------------------------------------------------------
    # Feature group 8: DNS features  [idx 35–38]
    # ------------------------------------------------------------------
    # dns_query_list: non-missing DNS query strings in this window
    f35_dns_query_count   = float(len(dns_query_list))
    f36_unique_dns_domains = float(len(set(dns_query_list)))
    # NXDomain: dns_rcode == '3' (NXDOMAIN in Zeek numeric encoding)
    f37_nxdomain_ratio    = _ratio(dns_rcode_list.count("3"), len(dns_rcode_list)) if dns_rcode_list else 0.0
    # DNS query entropy — high entropy = DGA activity (botnet indicator)
    f38_dns_query_entropy = _shannon_entropy(dns_query_list)

    # ------------------------------------------------------------------
    # Feature group 9: HTTP features  [idx 39–41]
    # ------------------------------------------------------------------
    http_flows = sum(1 for v in window_df.get("http_method", pd.Series(dtype=str))
                     if str(v).strip() not in _MISSING)
    f39_http_flow_ratio           = _ratio(http_flows, n)
    f40_mean_http_req_body_len    = _ratio(sum(http_req_list), len(http_req_list)) if http_req_list else 0.0
    f41_mean_http_resp_body_len   = _ratio(sum(http_resp_list), len(http_resp_list)) if http_resp_list else 0.0

    # ------------------------------------------------------------------
    # Feature group 10: SSL features  [idx 42–43]
    # ------------------------------------------------------------------
    f42_ssl_conn_ratio     = _ratio(len(ssl_version_list), n)
    f43_ssl_resumed_ratio  = _ratio(
        sum(1 for v in ssl_resumed_list if v.upper() in ("T", "TRUE", "1")),
        len(ssl_resumed_list)
    ) if ssl_resumed_list else 0.0

    # ------------------------------------------------------------------
    # Feature group 11: Missed-bytes ratio  [idx 44]
    # ------------------------------------------------------------------
    total_missed  = sum(missed_list)
    total_ip_bytes = sum(src_ip_bytes_list) + sum(dst_ip_bytes_list)
    f44_missed_bytes_ratio = _ratio(total_missed, total_ip_bytes)

    # ------------------------------------------------------------------
    # Feature group 12: Byte-size percentiles  [idx 45–50]
    # ------------------------------------------------------------------
    src_p25, src_p50, src_p75 = _percentiles(src_bytes_list)
    dst_p25, dst_p50, dst_p75 = _percentiles(dst_bytes_list)

    f45_src_bytes_p25 = src_p25
    f46_src_bytes_p50 = src_p50
    f47_src_bytes_p75 = src_p75
    f48_dst_bytes_p25 = dst_p25
    f49_dst_bytes_p50 = dst_p50
    f50_dst_bytes_p75 = dst_p75

    # ------------------------------------------------------------------
    # Feature group 13: Duration percentiles  [idx 51–53]
    # ------------------------------------------------------------------
    dur_p25, dur_p50, dur_p75 = _percentiles(duration_list)
    f51_duration_p25 = dur_p25
    f52_duration_p50 = dur_p50
    f53_duration_p75 = dur_p75

    # ------------------------------------------------------------------
    # Feature group 14: Inter-flow byte-gap stats  [idx 54–56]
    # ------------------------------------------------------------------
    # Sequential differences between consecutive src_bytes values.
    # Captures burstiness / periodicity patterns specific to each device class.
    if len(src_bytes_list) >= 2:
        gaps = [abs(src_bytes_list[i + 1] - src_bytes_list[i])
                for i in range(len(src_bytes_list) - 1)]
        f54_mean_byte_gap = float(np.mean(gaps))
        f55_std_byte_gap  = float(np.std(gaps))
        f56_max_byte_gap  = float(max(gaps))
    else:
        f54_mean_byte_gap = 0.0
        f55_std_byte_gap  = 0.0
        f56_max_byte_gap  = 0.0

    # ------------------------------------------------------------------
    # Feature group 15: Asymmetry / ratio features  [idx 57–59]
    # ------------------------------------------------------------------
    # bytes_out_in_ratio: measures whether device talks more than it listens
    # High ratio → data exfiltration signal; Low ratio → C2 command receiver
    f57_bytes_out_in_ratio = _ratio(total_src_bytes, total_dst_bytes)

    # pkts_out_in_ratio: packet-level asymmetry (DDoS often has high pkt ratio)
    f58_pkts_out_in_ratio = _ratio(total_src_pkts, total_dst_pkts)

    # fan_out_ratio: how many unique destinations per total flow
    # A device scanning ports will have high fan_out
    f59_fan_out_ratio = _ratio(float(unique_dst_ips + unique_dst_ports), float(n))

    # ------------------------------------------------------------------
    # Assemble final 60-dim vector
    # ------------------------------------------------------------------
    feature_vector = np.array([
        # Group 0: Volume (0–4)
        f00_flow_count, f01_total_src_bytes, f02_total_dst_bytes,
        f03_total_src_ip_bytes, f04_total_dst_ip_bytes,
        # Group 1: Packet stats (5–8)
        f05_mean_src_pkts, f06_std_src_pkts, f07_mean_dst_pkts, f08_std_dst_pkts,
        # Group 2: Duration stats (9–12)
        f09_mean_duration, f10_std_duration, f11_min_duration, f12_max_duration,
        # Group 3: Protocol ratios (13–16)
        f13_tcp_ratio, f14_udp_ratio, f15_icmp_ratio, f16_other_proto_ratio,
        # Group 4: Service ratios (17–20)
        f17_dns_service_ratio, f18_http_service_ratio,
        f19_ssl_service_ratio, f20_other_service_ratio,
        # Group 5: Conn-state ratios (21–27)
        f21_s0_ratio, f22_sf_ratio, f23_rej_ratio, f24_oth_ratio,
        f25_rsto_ratio, f26_shr_ratio, f27_other_state_ratio,
        # Group 6: Connectivity (28–30)
        f28_unique_dst_ips, f29_unique_dst_ports, f30_unique_src_ports,
        # Group 7: Port distribution (31–34)
        f31_mean_dst_port, f32_std_dst_port,
        f33_well_known_port_ratio, f34_ephemeral_port_ratio,
        # Group 8: DNS (35–38)
        f35_dns_query_count, f36_unique_dns_domains,
        f37_nxdomain_ratio, f38_dns_query_entropy,
        # Group 9: HTTP (39–41)
        f39_http_flow_ratio, f40_mean_http_req_body_len, f41_mean_http_resp_body_len,
        # Group 10: SSL (42–43)
        f42_ssl_conn_ratio, f43_ssl_resumed_ratio,
        # Group 11: Missed bytes (44)
        f44_missed_bytes_ratio,
        # Group 12: Byte percentiles (45–50)
        f45_src_bytes_p25, f46_src_bytes_p50, f47_src_bytes_p75,
        f48_dst_bytes_p25, f49_dst_bytes_p50, f50_dst_bytes_p75,
        # Group 13: Duration percentiles (51–53)
        f51_duration_p25, f52_duration_p50, f53_duration_p75,
        # Group 14: Inter-flow byte-gap stats (54–56)
        f54_mean_byte_gap, f55_std_byte_gap, f56_max_byte_gap,
        # Group 15: Asymmetry ratios (57–59)
        f57_bytes_out_in_ratio, f58_pkts_out_in_ratio, f59_fan_out_ratio,
    ], dtype=np.float32)

    # Sanity check — should never fire in production
    assert feature_vector.shape == (FEATURE_DIM,), (
        f"Feature vector has wrong shape: {feature_vector.shape}, expected ({FEATURE_DIM},)"
    )

    # Replace any NaN/Inf that slipped through (defensive — should not occur)
    if not np.all(np.isfinite(feature_vector)):
        n_bad = int(np.sum(~np.isfinite(feature_vector)))
        logger.warning(f"Replacing {n_bad} non-finite values in feature vector with 0.0")
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector


# ---------------------------------------------------------------------------
# Feature name registry — used by SHAP explainer and Grafana alerts
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # Group 0
    "flow_count", "total_src_bytes", "total_dst_bytes",
    "total_src_ip_bytes", "total_dst_ip_bytes",
    # Group 1
    "mean_src_pkts", "std_src_pkts", "mean_dst_pkts", "std_dst_pkts",
    # Group 2
    "mean_duration", "std_duration", "min_duration", "max_duration",
    # Group 3
    "tcp_ratio", "udp_ratio", "icmp_ratio", "other_proto_ratio",
    # Group 4
    "dns_service_ratio", "http_service_ratio",
    "ssl_service_ratio", "other_service_ratio",
    # Group 5
    "s0_ratio", "sf_ratio", "rej_ratio", "oth_ratio",
    "rsto_ratio", "shr_ratio", "other_state_ratio",
    # Group 6
    "unique_dst_ips", "unique_dst_ports", "unique_src_ports",
    # Group 7
    "mean_dst_port", "std_dst_port",
    "well_known_port_ratio", "ephemeral_port_ratio",
    # Group 8
    "dns_query_count", "unique_dns_domains",
    "nxdomain_ratio", "dns_query_entropy",
    # Group 9
    "http_flow_ratio", "mean_http_req_body_len", "mean_http_resp_body_len",
    # Group 10
    "ssl_conn_ratio", "ssl_resumed_ratio",
    # Group 11
    "missed_bytes_ratio",
    # Group 12
    "src_bytes_p25", "src_bytes_p50", "src_bytes_p75",
    "dst_bytes_p25", "dst_bytes_p50", "dst_bytes_p75",
    # Group 13
    "duration_p25", "duration_p50", "duration_p75",
    # Group 14
    "mean_byte_gap", "std_byte_gap", "max_byte_gap",
    # Group 15
    "bytes_out_in_ratio", "pkts_out_in_ratio", "fan_out_ratio",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, (
    f"FEATURE_NAMES length {len(FEATURE_NAMES)} != FEATURE_DIM {FEATURE_DIM}"
)
