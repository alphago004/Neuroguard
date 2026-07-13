"""
NEUROGUARD — IoT-23 Cross-Dataset Validation.

Purpose
-------
Train the NEUROGUARD Siamese encoder from scratch on IoT-23 benign traffic
and evaluate attack detection on the same dataset.  This is the cross-dataset
validation referenced in the paper's future-work section.

Three scenarios provide enough benign + attack data for per-device evaluation:
  CTU-IoT-Malware-Capture-34-1  192.168.1.195    Mirai  DDoS / C&C
  CTU-IoT-Malware-Capture-8-1   192.168.100.113  Mirai  C&C
  CTU-IoT-Malware-Capture-3-1   192.168.2.5      Mirai  PortScan / Attack

Two additional benign-heavy scenarios supplement training only:
  CTU-IoT-Malware-Capture-20-1  192.168.100.103  benign-dominant
  CTU-IoT-Malware-Capture-42-1  192.168.1.197    benign-dominant

Protocol (mirrors TON_IoT evaluation exactly):
  1. Parse IoT-23 conn.log.labeled files, rename columns to match extractor
  2. Separate benign-labeled flows (training/enrollment) from attack (test)
  3. Build 50-flow windows with 25-flow stride per device
  4. Split benign windows 80/20 per device → train_normal / test_normal
  5. Fit RobustScaler on train_normal ONLY
  6. Train Siamese encoder on IoT-23 train_normal pairs ONLY
  7. Enroll devices using test_normal
  8. Score attack windows → ROC-AUC, TPR, FPR, F1
  9. Save results to models/checkpoints/iot23_results.txt

Run with: python scripts/run_iot23_eval.py
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
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.extractor import extract_features, WINDOW_SIZE
from src.models.siamese import SiameseNetwork, ContrastiveLoss, build_model

import pandas as pd

# ---------------------------------------------------------------------------
# Scenario config
# ---------------------------------------------------------------------------
IOT23_DIR  = ROOT / "data" / "raw" / "iot23"
SAVE_DIR   = ROOT / "models" / "checkpoints"
RESULT_TXT = ROOT / "iot23_results.txt"

# Scenarios used for BOTH training AND attack evaluation
EVAL_SCENARIOS = [
    {
        "file":      IOT23_DIR / "malware_capture_34.conn.log.labeled",
        "device_ip": "192.168.1.195",
        "malware":   "Mirai (DDoS/C&C)",
        "scenario":  "CTU-IoT-Malware-Capture-34-1",
    },
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-8-1.conn.log.labeled",
        "device_ip": "192.168.100.113",
        "malware":   "Mirai (C&C)",
        "scenario":  "CTU-IoT-Malware-Capture-8-1",
    },
    {
        "file":      IOT23_DIR / "CTU-IoT-Malware-Capture-3-1.conn.log.labeled",
        "device_ip": "192.168.2.5",
        "malware":   "Mirai (PortScan/Attack)",
        "scenario":  "CTU-IoT-Malware-Capture-3-1",
    },
]

# Scenarios used for training only (too few attack windows for evaluation)
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

# Hyperparameters (slightly smaller than TON_IoT due to smaller dataset)
WINDOW_STRIDE  = 25
TRAIN_SPLIT    = 0.80
EPOCHS         = 80
BATCH_SIZE     = 64
LR             = 1e-3
PATIENCE       = 10
N_PAIRS        = 50_000
DEVICE_CAP     = 80
MAX_ATTACK_WIN = 2_000   # cap attack windows per device to limit eval time
MARGIN         = 3.0

# Column mapping: IoT-23 → TON_IoT extractor-compatible names
COLUMN_MAP = {
    "id.orig_h":    "src_ip",
    "id.orig_p":    "src_port",
    "id.resp_h":    "dst_ip",
    "id.resp_p":    "dst_port",
    "orig_bytes":   "src_bytes",
    "resp_bytes":   "dst_bytes",
    "orig_pkts":    "src_pkts",
    "orig_ip_bytes":"src_ip_bytes",
    "resp_pkts":    "dst_pkts",
    "resp_ip_bytes":"dst_ip_bytes",
    # proto, service, duration, conn_state, missed_bytes → same names, no rename
}


# ---------------------------------------------------------------------------
# IoT-23 file parser
# ---------------------------------------------------------------------------

def load_iot23_file(filepath: Path) -> pd.DataFrame:
    """Parse an IoT-23 conn.log.labeled file into a tidy DataFrame.

    Handles the two separator variants found in IoT-23:
      - Tab-only: first 22 fields all tab-separated
      - Mixed: first 21 fields tab-separated, last 2 space-separated
    """
    header: Optional[list[str]] = None
    rows: list[list[str]] = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#fields"):
                raw_fields = line[len("#fields"):].strip()
                # Normalize runs of ≥2 spaces to a tab (handles mixed format)
                normalized = re.sub(r" {2,}", "\t", raw_fields)
                header = [h.strip() for h in normalized.split("\t") if h.strip()]
            elif line.startswith("#"):
                continue
            elif header:
                normalized = re.sub(r" {2,}", "\t", line)
                parts = normalized.split("\t")
                # Pad or trim to header length
                while len(parts) < len(header):
                    parts.append("")
                rows.append(parts[: len(header)])

    if not header or not rows:
        logger.warning(f"No data parsed from {filepath}")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    return df


def prepare_iot23_df(df: pd.DataFrame, device_ip: str) -> pd.DataFrame:
    """Rename columns, filter to device IP, assign label column."""
    df = df.rename(columns=COLUMN_MAP)

    # Keep only flows where the known device is the source
    df = df[df["src_ip"] == device_ip].copy()
    if df.empty:
        return df

    # Determine label: benign row → type = 'normal', else type = attack string
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        logger.error("No 'label' column found")
        return pd.DataFrame()

    # Normalize label: 'Benign'/'benign' → 'normal', else keep as-is (attack)
    df["type"] = df[label_col].apply(
        lambda x: "normal" if "benign" in str(x).lower() else str(x).strip().lower()
    )
    df["label"] = (df["type"] != "normal").astype(int)

    # Cast numeric columns that the extractor needs
    for col in ["src_port", "dst_port", "src_bytes", "dst_bytes",
                "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
                "duration", "missed_bytes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Window record dataclass
# ---------------------------------------------------------------------------

@dataclass
class WinRecord:
    device_id:  str
    features:   np.ndarray   # shape (60,)
    label:      int          # 0 = normal, 1 = attack
    window_idx: int


def build_windows(df: pd.DataFrame, device_id: str,
                  window_size: int = WINDOW_SIZE,
                  stride: int = WINDOW_STRIDE) -> list[WinRecord]:
    """Slide a window over device flows and extract 60-dim feature vectors."""
    records = []
    n = len(df)
    if n < window_size:
        logger.debug(f"  {device_id}: only {n} flows, need {window_size} — skipping")
        return records

    n_windows = math.floor((n - window_size) / stride) + 1
    for w in range(n_windows):
        start = w * stride
        window_df = df.iloc[start: start + window_size].copy()
        # Label window: 1 if ANY flow is attack
        win_label = int((window_df["label"] > 0).any())
        try:
            features = extract_features(window_df)
        except Exception as e:
            logger.debug(f"  Feature extraction failed for window {w}: {e}")
            continue
        records.append(WinRecord(
            device_id=device_id,
            features=features,
            label=win_label,
            window_idx=w,
        ))
    return records


# ---------------------------------------------------------------------------
# Pair dataset for Siamese training
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Generates (anchor, sample, label) contrastive pairs."""

    def __init__(self,
                 normal_by_device: dict[str, list[WinRecord]],
                 n_pairs: int,
                 device_cap: int,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        devices = list(normal_by_device.keys())
        pairs: list[tuple[np.ndarray, np.ndarray, int]] = []

        for _ in range(n_pairs):
            # 50 / 50 positive / negative split
            if rng.random() < 0.5:
                # Positive pair (same device)
                dev = devices[rng.integers(len(devices))]
                wins = normal_by_device[dev]
                if len(wins) < 2:
                    continue
                idx = rng.choice(
                    min(len(wins), device_cap), 2, replace=False
                )
                pairs.append((wins[idx[0]].features, wins[idx[1]].features, 0))
            else:
                # Negative pair (different devices)
                if len(devices) < 2:
                    continue
                d1, d2 = rng.choice(len(devices), 2, replace=False)
                w1 = normal_by_device[devices[d1]]
                w2 = normal_by_device[devices[d2]]
                a = w1[rng.integers(min(len(w1), device_cap))]
                b = w2[rng.integers(min(len(w2), device_cap))]
                pairs.append((a.features, b.features, 1))

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b, lbl = self.pairs[idx]
        return (
            torch.from_numpy(a.astype(np.float32)),
            torch.from_numpy(b.astype(np.float32)),
            torch.tensor(lbl, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Enrollment (inline — mirrors enroll.py but without filesystem DNA cache)
# ---------------------------------------------------------------------------

K_SIGMA       = 2.5
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95


@dataclass
class DeviceDNA:
    device_id:          str
    centroid:           np.ndarray    # (64,) L2-normalised mean embedding
    embedding_distances: np.ndarray  # cosine distances of enrolled windows
    threshold_distance: float


@torch.no_grad()
def enroll_device(device_id: str,
                  windows: list[WinRecord],
                  model: SiameseNetwork,
                  scaler: RobustScaler,
                  pt_device: torch.device) -> Optional[DeviceDNA]:
    if not windows:
        return None

    X = scaler.transform(
        np.stack([w.features for w in windows])
    ).astype(np.float32)

    model.eval()
    t = torch.from_numpy(X).to(pt_device)
    embs = model.encoder(t).cpu().numpy()  # (N, 64)

    # L2-normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-8, None)

    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8

    # Cosine distance from each embedding to centroid
    dists = 1.0 - (embs @ centroid)

    threshold = float(np.clip(
        dists.mean() + K_SIGMA * dists.std(),
        MIN_THRESHOLD, MAX_THRESHOLD
    ))

    return DeviceDNA(
        device_id=device_id,
        centroid=centroid,
        embedding_distances=dists,
        threshold_distance=threshold,
    )


@torch.no_grad()
def score_windows(windows: list[WinRecord],
                  dna: DeviceDNA,
                  model: SiameseNetwork,
                  scaler: RobustScaler,
                  pt_device: torch.device) -> list[float]:
    """Return cosine-distance anomaly scores for each window."""
    if not windows:
        return []

    X = scaler.transform(
        np.stack([w.features for w in windows])
    ).astype(np.float32)

    model.eval()
    t  = torch.from_numpy(X).to(pt_device)
    embs = model.encoder(t).cpu().numpy()

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-8, None)

    distances = 1.0 - (embs @ dna.centroid)
    return distances.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── PyTorch device ─────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        pt_device = torch.device("mps")
    elif torch.cuda.is_available():
        pt_device = torch.device("cuda")
    else:
        pt_device = torch.device("cpu")
    logger.info(f"PyTorch device: {pt_device}")

    # ── 1. Load and parse all scenarios ────────────────────────────────────
    logger.info("Loading IoT-23 scenario files…")

    # normal_by_device: all benign windows (for training + enrollment)
    normal_by_device: dict[str, list[WinRecord]] = {}
    # attack_by_device: attack windows (for evaluation only)
    attack_by_device: dict[str, list[WinRecord]] = {}
    scenario_info: dict[str, dict] = {}

    all_scenarios = EVAL_SCENARIOS + TRAIN_ONLY_SCENARIOS
    for sc in all_scenarios:
        fpath     = sc["file"]
        device_ip = sc["device_ip"]

        if not fpath.exists():
            logger.warning(f"  File not found: {fpath} — skipping")
            continue

        logger.info(f"  Loading {fpath.name} (device {device_ip})…")
        raw_df = load_iot23_file(fpath)
        if raw_df.empty:
            logger.warning(f"  Empty parse result for {fpath.name}")
            continue

        df = prepare_iot23_df(raw_df, device_ip)
        if df.empty:
            logger.warning(f"  No rows for device {device_ip} in {fpath.name}")
            continue

        benign_df = df[df["label"] == 0].reset_index(drop=True)
        attack_df = df[df["label"] == 1].reset_index(drop=True)

        logger.info(
            f"    benign flows: {len(benign_df)}  "
            f"attack flows: {len(attack_df)}"
        )

        # Build windows from benign flows
        all_benign_wins = build_windows(benign_df, device_ip)
        if not all_benign_wins:
            logger.warning(f"  No benign windows for {device_ip} — skipping")
            continue

        normal_by_device[device_ip] = all_benign_wins
        logger.info(f"    benign windows: {len(all_benign_wins)}")

        # Build windows from attack flows (eval scenarios only, capped)
        if sc in EVAL_SCENARIOS and len(attack_df) >= WINDOW_SIZE:
            atk_wins = build_windows(attack_df, device_ip)
            # Cap to avoid imbalance and long eval time
            if len(atk_wins) > MAX_ATTACK_WIN:
                atk_wins = atk_wins[:MAX_ATTACK_WIN]
            attack_by_device[device_ip] = atk_wins
            logger.info(f"    attack windows: {len(atk_wins)}")
            scenario_info[device_ip] = sc

    if not normal_by_device:
        logger.error("No normal windows loaded — aborting")
        return

    eval_devices = [ip for ip in attack_by_device if ip in normal_by_device]
    if not eval_devices:
        logger.error("No evaluation devices available — aborting")
        return

    # ── 2. Train / test split (80/20 per device, benign only) ──────────────
    logger.info("Splitting benign windows into train/test per device…")
    train_normal_by_device: dict[str, list[WinRecord]] = {}
    test_normal_by_device:  dict[str, list[WinRecord]] = {}

    for dev, wins in normal_by_device.items():
        n_train = max(1, int(len(wins) * TRAIN_SPLIT))
        train_normal_by_device[dev] = wins[:n_train]
        test_normal_by_device[dev]  = wins[n_train:]
        logger.info(
            f"  {dev}: {len(wins)} benign windows → "
            f"{n_train} train / {len(wins) - n_train} test"
        )

    # ── 3. Fit RobustScaler on train_normal ────────────────────────────────
    logger.info("Fitting RobustScaler on IoT-23 train_normal windows…")
    all_train = [w for wins in train_normal_by_device.values() for w in wins]
    X_train_raw = np.stack([w.features for w in all_train])
    scaler = RobustScaler()
    scaler.fit(X_train_raw)
    logger.info(f"  Scaler fitted on {len(all_train)} windows")

    with open(SAVE_DIR / "scaler_iot23.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── 4. Train Siamese model ─────────────────────────────────────────────
    logger.info(f"Training Siamese encoder on IoT-23 normal traffic…")
    model, loss_fn = build_model(transformer_layers=1, margin=MARGIN)
    model = model.to(pt_device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    # Scale train_normal for pair generation
    scaled_train: dict[str, list[WinRecord]] = {}
    for dev, wins in train_normal_by_device.items():
        X = scaler.transform(np.stack([w.features for w in wins])).astype(np.float32)
        scaled_train[dev] = [
            WinRecord(w.device_id, X[i], w.label, w.window_idx)
            for i, w in enumerate(wins)
        ]

    # Validation pairs (from test_normal)
    scaled_test: dict[str, list[WinRecord]] = {}
    for dev, wins in test_normal_by_device.items():
        if not wins:
            continue
        X = scaler.transform(np.stack([w.features for w in wins])).astype(np.float32)
        scaled_test[dev] = [
            WinRecord(w.device_id, X[i], w.label, w.window_idx)
            for i, w in enumerate(wins)
        ]

    val_pairs  = PairDataset(scaled_test, n_pairs=N_PAIRS, device_cap=DEVICE_CAP, seed=0)
    val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improv = 0

    logger.info(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Status'}")
    logger.info("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        epoch_pairs  = PairDataset(scaled_train, n_pairs=N_PAIRS,
                                   device_cap=DEVICE_CAP, seed=epoch * 7)
        train_loader = DataLoader(epoch_pairs, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0)

        model.train()
        running_loss = 0.0
        for anchor, pair, labels in train_loader:
            anchor = anchor.to(pt_device)
            pair   = pair.to(pt_device)
            labels = labels.to(pt_device)
            optimizer.zero_grad()
            emb_a, emb_b = model(anchor, pair)
            dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
            loss = loss_fn(dist, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for anchor, pair, labels in val_loader:
                anchor = anchor.to(pt_device)
                pair   = pair.to(pt_device)
                labels = labels.to(pt_device)
                emb_a, emb_b = model(anchor, pair)
                dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
                loss = loss_fn(dist, labels)
                val_running += loss.item()
        val_loss = val_running / len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improv = 0
            torch.save(model.state_dict(), SAVE_DIR / "best_model_iot23.pt")
            status = "saved"
        else:
            epochs_no_improv += 1
            status = f"no improv {epochs_no_improv}/{PATIENCE}"

        logger.info(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  {status}")

        if epochs_no_improv >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best checkpoint
    model.load_state_dict(
        torch.load(SAVE_DIR / "best_model_iot23.pt", map_location=pt_device)
    )
    logger.info(f"Best model at epoch {best_epoch} (val_loss={best_val_loss:.6f})")

    # ── 5. Enroll + score ─────────────────────────────────────────────────
    logger.info("Enrolling devices and scoring attacks…")

    all_scores_normal: list[float] = []
    all_scores_attack:  list[float] = []
    per_device_results: dict[str, dict] = {}

    for dev in eval_devices:
        test_normal_wins = test_normal_by_device.get(dev, [])
        attack_wins      = attack_by_device.get(dev, [])

        if not test_normal_wins:
            logger.warning(f"  {dev}: no test_normal windows — skipping")
            continue

        if not attack_wins:
            logger.warning(f"  {dev}: no attack windows — skipping")
            continue

        dna = enroll_device(dev, test_normal_wins, model, scaler, pt_device)
        if dna is None:
            logger.warning(f"  {dev}: enrollment failed — skipping")
            continue

        logger.info(
            f"  {dev}: enrolled on {len(test_normal_wins)} windows  "
            f"threshold={dna.threshold_distance:.4f}"
        )

        scores_n = score_windows(test_normal_wins, dna, model, scaler, pt_device)
        scores_a  = score_windows(attack_wins,      dna, model, scaler, pt_device)

        all_scores_normal.extend(scores_n)
        all_scores_attack.extend(scores_a)

        # Per-device confusion at threshold
        tp = sum(1 for s in scores_a if s >= dna.threshold_distance)
        fp = sum(1 for s in scores_n if s >= dna.threshold_distance)
        fn = len(scores_a) - tp
        tn = len(scores_n) - fp

        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2*prec*tpr / (prec+tpr) if (prec+tpr) > 0 else 0.0

        per_device_results[dev] = {
            "malware":   scenario_info[dev]["malware"],
            "n_normal":  len(scores_n),
            "n_attack":  len(scores_a),
            "tpr": tpr, "fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    if not all_scores_normal or not all_scores_attack:
        logger.error("No scores collected — cannot compute ROC-AUC")
        return

    # ── 6. Aggregate metrics ───────────────────────────────────────────────
    labels_all = [0] * len(all_scores_normal) + [1] * len(all_scores_attack)
    scores_all = all_scores_normal + all_scores_attack
    roc_auc    = roc_auc_score(labels_all, scores_all)

    sn = np.array(all_scores_normal)
    sa = np.array(all_scores_attack)
    score_gap = float(np.percentile(sa, 5)) - float(np.percentile(sn, 95))

    total_time = time.time() - t_start

    # ── 7. Print results ───────────────────────────────────────────────────
    W = 72
    print(f"\n{'═'*W}")
    print(f"  NEUROGUARD — IoT-23 CROSS-DATASET VALIDATION")
    print(f"{'═'*W}")
    print(f"  ROC-AUC (aggregate, 3 devices)  : {roc_auc:.4f}")
    print(f"  Score gap (attack p5 - normal p95): {score_gap:+.4f}")
    print(f"  Best training epoch              : {best_epoch}")
    print(f"  Total time                       : {total_time/60:.1f} min")
    print(f"\n  {'Device':<18} {'Malware':<28} {'N-win':>6} {'A-win':>6} "
          f"{'TPR':>7} {'FPR':>7} {'F1':>7}")
    print(f"  {'─'*18:<18} {'─'*28:<28} {'─'*6:>6} {'─'*6:>6} "
          f"{'─'*7:>7} {'─'*7:>7} {'─'*7:>7}")
    for dev, m in per_device_results.items():
        print(
            f"  {dev:<18} {m['malware']:<28} {m['n_normal']:>6} {m['n_attack']:>6} "
            f"{m['tpr']:>6.1%} {m['fpr']:>6.1%} {m['f1']:>7.4f}"
        )
    print(f"{'═'*W}\n")

    # ── 8. Save results ───────────────────────────────────────────────────
    with open(RESULT_TXT, "w") as f:
        f.write("NEUROGUARD IoT-23 Cross-Dataset Validation\n")
        f.write(f"roc_auc: {roc_auc:.4f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"n_normal_windows: {len(all_scores_normal)}\n")
        f.write(f"n_attack_windows: {len(all_scores_attack)}\n")
        f.write(f"total_time_s: {total_time:.1f}\n")
        for dev, m in per_device_results.items():
            f.write(f"\n{dev} ({m['malware']}):\n")
            f.write(f"  tpr={m['tpr']:.4f}  fpr={m['fpr']:.4f}  f1={m['f1']:.4f}\n")
            f.write(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}\n")
    logger.info(f"Results saved → {RESULT_TXT}")


if __name__ == "__main__":
    main()
