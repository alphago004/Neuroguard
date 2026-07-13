"""
NEUROGUARD — Zero-Shot Cross-Dataset Transfer (TON_IoT → IoT-23).

What this tests
---------------
The TON_IoT-trained encoder (best_model.pt) is applied to IoT-23 attack
detection WITHOUT any retraining.  No IoT-23 data was seen during encoder
training.  This is the strongest form of generalization evidence: do the
behavioral representations learned from one network environment transfer
to a completely different one?

Protocol
--------
1. Load best_model.pt  (TON_IoT encoder, 609k params, frozen weights)
2. Load scaler_iot23.pkl  (RobustScaler fitted on IoT-23 benign data only)
   Using the IoT-23 scaler — not the TON_IoT scaler — is essential: IoT-23
   feature distributions differ from TON_IoT (different device classes,
   different byte/timing ranges, no DNS/HTTP/SSL features).  The scaler
   normalizes IoT-23 features to a comparable scale before feeding them to
   the encoder.  The encoder weights remain untouched — this is not training.
3. Parse IoT-23 conn.log.labeled files (same parser as run_iot23_eval.py)
4. Build 50-flow windows for 3 eval devices + 2 training-only devices
5. Enroll each eval device using its test_normal windows through the
   TON_IoT encoder
6. Score attack windows → cosine distance from enrolled centroid
7. Report ROC-AUC, TPR, FPR, F1 per device and aggregate

Key difference from run_iot23_eval.py
--------------------------------------
run_iot23_eval.py  trains a fresh Siamese model on IoT-23 benign data.
This script         uses the frozen TON_IoT encoder — zero training on IoT-23.

Run with: python scripts/run_zero_shot_iot23.py
Results → zero_shot_iot23_results.txt
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.extractor import extract_features, WINDOW_SIZE
from src.models.siamese import build_model

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IOT23_DIR    = ROOT / "data" / "raw" / "iot23"
CHECKPOINT   = ROOT / "models" / "checkpoints" / "best_model.pt"      # TON_IoT encoder
SCALER_IOT23 = ROOT / "models" / "checkpoints" / "scaler_iot23.pkl"   # IoT-23 scaler
RESULT_TXT   = ROOT / "zero_shot_iot23_results.txt"

# ---------------------------------------------------------------------------
# Scenario config  (identical to run_iot23_eval.py)
# ---------------------------------------------------------------------------
EVAL_SCENARIOS = [
    {
        "file":      IOT23_DIR / "malware_capture_34.conn.log.labeled",
        "device_ip": "192.168.1.195",
        "malware":   "Mirai (DDoS/C&C)",
        "scenario":  "CTU-34-1",
    },
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-8-1.conn.log.labeled",
        "device_ip": "192.168.100.113",
        "malware":   "Mirai (C&C)",
        "scenario":  "CTU-8-1",
    },
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-3-1.conn.log.labeled",
        "device_ip": "192.168.2.5",
        "malware":   "Mirai (PortScan)",
        "scenario":  "CTU-3-1",
    },
]

TRAIN_ONLY_SCENARIOS = [
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-20-1.conn.log.labeled",
        "device_ip": "192.168.100.103",
    },
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-42-1.conn.log.labeled",
        "device_ip": "192.168.1.197",
    },
]

WINDOW_STRIDE  = 25
TRAIN_SPLIT    = 0.80
MAX_ATTACK_WIN = 2_000

# Enrollment parameters (same as src/detection/enroll.py)
K_SIGMA       = 2.5
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95

COLUMN_MAP = {
    "id.orig_h":     "src_ip",
    "id.orig_p":     "src_port",
    "id.resp_h":     "dst_ip",
    "id.resp_p":     "dst_port",
    "orig_bytes":    "src_bytes",
    "resp_bytes":    "dst_bytes",
    "orig_pkts":     "src_pkts",
    "orig_ip_bytes": "src_ip_bytes",
    "resp_pkts":     "dst_pkts",
    "resp_ip_bytes": "dst_ip_bytes",
}


# ---------------------------------------------------------------------------
# IoT-23 parser  (proven in run_iot23_eval.py)
# ---------------------------------------------------------------------------

def load_iot23_file(filepath: Path) -> pd.DataFrame:
    """Parse IoT-23 conn.log.labeled — handles tab-only and mixed separators."""
    header: Optional[list[str]] = None
    rows:   list[list[str]]     = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#fields"):
                raw_fields = line[len("#fields"):].strip()
                normalized = re.sub(r" {2,}", "\t", raw_fields)
                header = [h.strip() for h in normalized.split("\t") if h.strip()]
            elif line.startswith("#"):
                continue
            elif header:
                normalized = re.sub(r" {2,}", "\t", line)
                parts = normalized.split("\t")
                while len(parts) < len(header):
                    parts.append("")
                rows.append(parts[: len(header)])

    if not header or not rows:
        logger.warning(f"No data parsed from {filepath.name}")
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)


def prepare_iot23_df(df: pd.DataFrame, device_ip: str) -> pd.DataFrame:
    """Rename columns, filter to device IP, assign binary label."""
    df = df.rename(columns=COLUMN_MAP)
    df = df[df["src_ip"] == device_ip].copy()
    if df.empty:
        return df

    if "label" not in df.columns:
        logger.error(f"No 'label' column for {device_ip}")
        return pd.DataFrame()

    df["type"] = df["label"].apply(
        lambda x: "normal" if "benign" in str(x).lower() else str(x).strip().lower()
    )
    df["label"] = (df["type"] != "normal").astype(int)

    for col in ["src_port", "dst_port", "src_bytes", "dst_bytes",
                "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
                "duration", "missed_bytes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Window record
# ---------------------------------------------------------------------------

@dataclass
class WinRecord:
    device_id:  str
    features:   np.ndarray   # (60,) float32 — NOT yet scaled
    label:      int          # 0 = normal, 1 = attack
    window_idx: int


def build_windows(df: pd.DataFrame, device_id: str,
                  window_size: int = WINDOW_SIZE,
                  stride: int = WINDOW_STRIDE) -> list[WinRecord]:
    records = []
    n = len(df)
    if n < window_size:
        logger.debug(f"  {device_id}: {n} flows < {window_size} — skipping")
        return records

    n_windows = math.floor((n - window_size) / stride) + 1
    for w in range(n_windows):
        start      = w * stride
        window_df  = df.iloc[start: start + window_size].copy()
        win_label  = int((window_df["label"] > 0).any())
        try:
            features = extract_features(window_df)
        except Exception as e:
            logger.debug(f"  Window {w} extraction failed: {e}")
            continue
        records.append(WinRecord(device_id, features, win_label, w))
    return records


# ---------------------------------------------------------------------------
# Enrollment + scoring (inline — same math as src/detection/enroll.py)
# ---------------------------------------------------------------------------

@dataclass
class DeviceDNA:
    device_id:           str
    centroid:            np.ndarray   # (64,) L2-normalised
    embedding_distances: np.ndarray   # cosine distances of enrolled windows
    threshold_distance:  float


@torch.no_grad()
def enroll_device(device_id: str,
                  windows: list[WinRecord],
                  model,
                  scaler: RobustScaler,
                  pt_device: torch.device) -> Optional[DeviceDNA]:
    if not windows:
        return None

    X = scaler.transform(
        np.stack([w.features for w in windows])
    ).astype(np.float32)

    model.eval()
    embs = model.encoder(torch.from_numpy(X).to(pt_device)).cpu().numpy()

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-8, None)

    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8

    dists     = 1.0 - (embs @ centroid)
    threshold = float(np.clip(
        dists.mean() + K_SIGMA * dists.std(),
        MIN_THRESHOLD, MAX_THRESHOLD,
    ))

    return DeviceDNA(device_id, centroid, dists, threshold)


@torch.no_grad()
def score_windows(windows: list[WinRecord],
                  dna: DeviceDNA,
                  model,
                  scaler: RobustScaler,
                  pt_device: torch.device) -> list[float]:
    if not windows:
        return []

    X = scaler.transform(
        np.stack([w.features for w in windows])
    ).astype(np.float32)

    model.eval()
    embs = model.encoder(torch.from_numpy(X).to(pt_device)).cpu().numpy()

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-8, None)

    return (1.0 - (embs @ dna.centroid)).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        pt_device = torch.device("mps")
    elif torch.cuda.is_available():
        pt_device = torch.device("cuda")
    else:
        pt_device = torch.device("cpu")
    logger.info(f"Device: {pt_device}")

    # ── 1. Load TON_IoT encoder (frozen — no training) ────────────────────────
    logger.info(f"Loading TON_IoT encoder from {CHECKPOINT.name}…")
    model, _ = build_model(transformer_layers=1, margin=3.0)
    state = torch.load(CHECKPOINT, map_location=pt_device)
    model.load_state_dict(state)
    model = model.to(pt_device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Encoder loaded: {n_params:,} params — FROZEN, no training")

    # ── 2. Load IoT-23 scaler ─────────────────────────────────────────────────
    logger.info(f"Loading IoT-23 scaler from {SCALER_IOT23.name}…")
    with open(SCALER_IOT23, "rb") as f:
        scaler: RobustScaler = pickle.load(f)
    logger.info("  Scaler loaded — fitted on IoT-23 benign windows only")

    # ── 3. Parse IoT-23 files ─────────────────────────────────────────────────
    logger.info("Parsing IoT-23 scenario files…")

    normal_by_device: dict[str, list[WinRecord]] = {}
    attack_by_device: dict[str, list[WinRecord]] = {}
    scenario_info:    dict[str, dict]             = {}

    all_scenarios = EVAL_SCENARIOS + TRAIN_ONLY_SCENARIOS
    for sc in all_scenarios:
        fpath     = sc["file"]
        device_ip = sc["device_ip"]

        if not fpath.exists():
            logger.warning(f"  Missing: {fpath.name} — skipping")
            continue

        logger.info(f"  {fpath.name}  [{device_ip}]")
        raw_df = load_iot23_file(fpath)
        if raw_df.empty:
            continue

        df = prepare_iot23_df(raw_df, device_ip)
        if df.empty:
            logger.warning(f"    No rows for {device_ip}")
            continue

        benign_df = df[df["label"] == 0].reset_index(drop=True)
        attack_df = df[df["label"] == 1].reset_index(drop=True)
        logger.info(f"    benign flows: {len(benign_df)}  attack flows: {len(attack_df)}")

        benign_wins = build_windows(benign_df, device_ip)
        if not benign_wins:
            logger.warning(f"    No benign windows for {device_ip}")
            continue

        normal_by_device[device_ip] = benign_wins
        logger.info(f"    benign windows: {len(benign_wins)}")

        if sc in EVAL_SCENARIOS and len(attack_df) >= WINDOW_SIZE:
            atk_wins = build_windows(attack_df, device_ip)
            if len(atk_wins) > MAX_ATTACK_WIN:
                atk_wins = atk_wins[:MAX_ATTACK_WIN]
            attack_by_device[device_ip] = atk_wins
            logger.info(f"    attack windows: {len(atk_wins)}")
            scenario_info[device_ip] = sc

    eval_devices = [ip for ip in attack_by_device if ip in normal_by_device]
    if not eval_devices:
        logger.error("No evaluation devices — aborting")
        return

    # ── 4. 80/20 split of benign windows (same protocol as Phase 2) ───────────
    logger.info("Splitting benign windows 80/20 per device…")
    test_normal_by_device: dict[str, list[WinRecord]] = {}

    for dev, wins in normal_by_device.items():
        n_train = max(1, int(len(wins) * TRAIN_SPLIT))
        test_normal_by_device[dev] = wins[n_train:]
        logger.info(f"  {dev}: {len(wins)} benign → "
                    f"{n_train} train (unused) / {len(wins)-n_train} enroll")

    # ── 5. Enroll + score through TON_IoT encoder ─────────────────────────────
    logger.info("Enrolling devices with TON_IoT encoder and scoring attacks…")

    all_scores_normal: list[float] = []
    all_scores_attack:  list[float] = []
    per_device_results: dict[str, dict] = {}

    for dev in eval_devices:
        test_normal_wins = test_normal_by_device.get(dev, [])
        attack_wins      = attack_by_device.get(dev, [])

        if not test_normal_wins or not attack_wins:
            logger.warning(f"  {dev}: insufficient windows — skipping")
            continue

        dna = enroll_device(dev, test_normal_wins, model, scaler, pt_device)
        if dna is None:
            logger.warning(f"  {dev}: enrollment failed — skipping")
            continue

        logger.info(f"  {dev}: enrolled on {len(test_normal_wins)} windows  "
                    f"threshold={dna.threshold_distance:.4f}")

        scores_n = score_windows(test_normal_wins, dna, model, scaler, pt_device)
        scores_a  = score_windows(attack_wins,      dna, model, scaler, pt_device)

        all_scores_normal.extend(scores_n)
        all_scores_attack.extend(scores_a)

        tp = sum(1 for s in scores_a if s >= dna.threshold_distance)
        fp = sum(1 for s in scores_n if s >= dna.threshold_distance)
        fn = len(scores_a) - tp
        tn = len(scores_n) - fp

        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0

        per_device_results[dev] = {
            "malware":  scenario_info[dev]["malware"],
            "n_normal": len(scores_n),
            "n_attack": len(scores_a),
            "threshold": dna.threshold_distance,
            "tpr": tpr, "fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "normal_mean": float(np.mean(scores_n)),
            "attack_mean": float(np.mean(scores_a)),
        }

    if not all_scores_normal or not all_scores_attack:
        logger.error("No scores collected — aborting")
        return

    # ── 6. Aggregate metrics ──────────────────────────────────────────────────
    labels_all = [0] * len(all_scores_normal) + [1] * len(all_scores_attack)
    scores_all = all_scores_normal + all_scores_attack
    roc_auc    = roc_auc_score(labels_all, scores_all)

    sn = np.array(all_scores_normal)
    sa = np.array(all_scores_attack)
    score_gap  = float(np.percentile(sa, 5)) - float(np.percentile(sn, 95))
    total_time = time.time() - t_start

    # ── 7. Print results ──────────────────────────────────────────────────────
    W = 76
    print(f"\n{'═'*W}")
    print(f"  NEUROGUARD — ZERO-SHOT CROSS-DATASET TRANSFER (TON_IoT → IoT-23)")
    print(f"  Encoder: TON_IoT best_model.pt  |  Scaler: IoT-23 benign only")
    print(f"  No IoT-23 encoder training whatsoever")
    print(f"{'═'*W}")
    print(f"  Aggregate ROC-AUC              : {roc_auc:.4f}")
    print(f"  Score gap (attack p5 - norm p95): {score_gap:+.4f}")
    print(f"  Normal windows scored          : {len(all_scores_normal)}")
    print(f"  Attack windows scored          : {len(all_scores_attack)}")
    print(f"  Elapsed                        : {total_time/60:.1f} min")
    print(f"\n  {'Device':<18} {'Malware':<26} {'Enroll':>6} {'Attack':>6} "
          f"{'Thresh':>7} {'TPR':>7} {'FPR':>7} {'F1':>7}")
    print(f"  {'─'*18:<18} {'─'*26:<26} {'─'*6:>6} {'─'*6:>6} "
          f"{'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7}")
    for dev, m in per_device_results.items():
        print(
            f"  {dev:<18} {m['malware']:<26} {m['n_normal']:>6} {m['n_attack']:>6} "
            f"{m['threshold']:>7.4f} {m['tpr']:>6.1%} {m['fpr']:>6.1%} {m['f1']:>7.4f}"
        )
    print(f"\n  Score distributions (normal / attack):")
    for dev, m in per_device_results.items():
        print(f"  {dev}: normal_mean={m['normal_mean']:.4f}  "
              f"attack_mean={m['attack_mean']:.4f}  "
              f"gap={m['attack_mean']-m['normal_mean']:+.4f}")
    print(f"{'═'*W}\n")

    # ── 8. Save results ───────────────────────────────────────────────────────
    with open(RESULT_TXT, "w") as f:
        f.write("NEUROGUARD Zero-Shot Cross-Dataset Transfer (TON_IoT → IoT-23)\n")
        f.write(f"encoder: best_model.pt (TON_IoT, no IoT-23 training)\n")
        f.write(f"scaler: scaler_iot23.pkl (IoT-23 benign only)\n")
        f.write(f"roc_auc: {roc_auc:.4f}\n")
        f.write(f"score_gap: {score_gap:+.4f}\n")
        f.write(f"n_normal_windows: {len(all_scores_normal)}\n")
        f.write(f"n_attack_windows: {len(all_scores_attack)}\n")
        f.write(f"total_time_s: {total_time:.1f}\n")
        for dev, m in per_device_results.items():
            f.write(f"\n{dev} ({m['malware']}):\n")
            f.write(f"  tpr={m['tpr']:.4f}  fpr={m['fpr']:.4f}  f1={m['f1']:.4f}\n")
            f.write(f"  threshold={m['threshold']:.4f}\n")
            f.write(f"  normal_mean={m['normal_mean']:.4f}  "
                    f"attack_mean={m['attack_mean']:.4f}\n")
            f.write(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}\n")

    logger.info(f"Results saved → {RESULT_TXT}")


if __name__ == "__main__":
    main()
