"""
NEUROGUARD — Fair per-device baseline comparison.

THE FIX:
  The original run_baselines.py trained one global model on all 1360 normal
  windows pooled from all 16 devices. NEUROGUARD is per-device. That is an
  unfair comparison — the baselines were given a structural disadvantage.

  This script gives every baseline the same per-device treatment:
    - For each device, train a SEPARATE model on that device's normal
      training windows only.
    - Score that device's test_normal and attack windows with its own model.
    - Aggregate scores across all devices → compute ROC-AUC and confusion
      matrices exactly as run_evaluation.py does for NEUROGUARD.

Baselines:
  1. IsolationForest  (per-device, contamination=auto)
  2. OneClassSVM      (per-device, nu=0.1, rbf kernel)
  3. Autoencoder      (per-device, 60→32→16→32→60, MSE threshold @ p95)

Output:
  - Console table ready to paste into paper Table IV
  - Saves results to models/checkpoints/baselines_per_device.pkl
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from loguru import logger

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.dataset import WindowDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH  = ROOT / "data" / "processed" / "window_dataset.pkl"
SCALER_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"
OUT_PATH    = ROOT / "models" / "checkpoints" / "baselines_per_device.pkl"

# ---------------------------------------------------------------------------
# NEUROGUARD reference numbers (from run_evaluation.py)
# ---------------------------------------------------------------------------
NEUROGUARD = {
    "roc_auc": 0.9402,
    "tpr_25":  0.9833,
    "fpr_25":  0.0634,
    "f1_25":   0.9786,
    "tpr_30":  0.9762,
    "fpr_30":  0.0317,
    "f1_30":   0.9814,
}

# Minimum normal windows a device must have to train a per-device model.
# Devices with fewer windows are skipped (same policy as enrollment in enroll.py).
MIN_TRAIN_WINDOWS = 5


# ---------------------------------------------------------------------------
# Autoencoder definition (same architecture as run_baselines.py)
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Symmetric autoencoder: 60 → 32 → 16 → 32 → 60."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 60),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _train_autoencoder(X_train: np.ndarray, ae_device: torch.device) -> Autoencoder:
    """Train one Autoencoder on a single device's normal windows."""
    AE_EPOCHS   = 100
    AE_BATCH    = min(32, max(4, len(X_train) // 4))  # small batch for small devices
    AE_LR       = 1e-3
    AE_PATIENCE = 10

    ae = Autoencoder().to(ae_device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=AE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AE_EPOCHS)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(ae_device)

    # 80/20 train/val split within this device's windows
    n_val = max(1, int(len(X_t) * 0.20))
    X_ae_val   = X_t[-n_val:]
    X_ae_train = X_t[:-n_val]

    if len(X_ae_train) == 0:
        # Edge case: only 1 window, train and val on same data
        X_ae_train = X_t
        X_ae_val   = X_t

    train_ds = torch.utils.data.TensorDataset(X_ae_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=AE_BATCH, shuffle=True)

    best_val   = float("inf")
    patience_n = 0
    best_state = None

    for _ in range(1, AE_EPOCHS + 1):
        ae.train()
        for (batch,) in train_dl:
            optimizer.zero_grad()
            loss = criterion(ae(batch), batch)
            loss.backward()
            optimizer.step()

        ae.eval()
        with torch.no_grad():
            val_loss = criterion(ae(X_ae_val), X_ae_val).item()
        scheduler.step()

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            patience_n = 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            patience_n += 1

        if patience_n >= AE_PATIENCE:
            break

    if best_state is not None:
        ae.load_state_dict(best_state)
    ae.eval()
    return ae


def _ae_recon_errors(ae: Autoencoder,
                     X: np.ndarray,
                     ae_device: torch.device) -> np.ndarray:
    t = torch.from_numpy(X.astype(np.float32)).to(ae_device)
    with torch.no_grad():
        recon = ae(t)
    return ((t - recon) ** 2).mean(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Confusion stats helper
# ---------------------------------------------------------------------------

def _confusion(scores_normal: list[float],
               scores_attack: list[float],
               threshold: float) -> dict:
    sn = np.array(scores_normal)
    sa = np.array(scores_attack)

    tp = int((sa >= threshold).sum())
    fn = int((sa <  threshold).sum())
    fp = int((sn >= threshold).sum())
    tn = int((sn <  threshold).sum())

    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2*prec*tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                tpr=tpr, fpr=fpr, precision=prec, f1=f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading WindowDataset cache…")
    window_ds = WindowDataset.load(CACHE_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler: RobustScaler = pickle.load(f)

    # Group records by device
    train_by_device:  dict[str, list] = defaultdict(list)
    test_n_by_device: dict[str, list] = defaultdict(list)
    attack_by_device: dict[str, list] = defaultdict(list)

    for r in window_ds.train_normal:
        train_by_device[r.device_id].append(r)
    for r in window_ds.test_normal:
        test_n_by_device[r.device_id].append(r)
    for r in window_ds.attack_records:
        attack_by_device[r.device_id].append(r)

    all_devices = sorted(train_by_device.keys())
    logger.info(f"Devices with training windows: {len(all_devices)}")
    for dev in all_devices:
        logger.info(
            f"  {dev}  train_normal={len(train_by_device[dev])}  "
            f"test_normal={len(test_n_by_device[dev])}  "
            f"attack={len(attack_by_device[dev])}"
        )

    # ── 2. PyTorch device ─────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        ae_device = torch.device("mps")
    elif torch.cuda.is_available():
        ae_device = torch.device("cuda")
    else:
        ae_device = torch.device("cpu")
    logger.info(f"Autoencoder training device: {ae_device}")

    # ── 3. Per-device training + scoring ──────────────────────────────────────
    # Collect per-window scores + ground-truth labels across all devices
    all_scores_if_normal:  list[float] = []
    all_scores_if_attack:  list[float] = []
    all_scores_svm_normal: list[float] = []
    all_scores_svm_attack: list[float] = []
    all_scores_ae_normal:  list[float] = []
    all_scores_ae_attack:  list[float] = []

    skipped = []

    for dev in all_devices:
        train_records = train_by_device[dev]
        test_records  = test_n_by_device.get(dev, [])
        atk_records   = attack_by_device.get(dev, [])

        if len(train_records) < MIN_TRAIN_WINDOWS:
            logger.warning(
                f"  {dev}: only {len(train_records)} train windows "
                f"(< {MIN_TRAIN_WINDOWS}) — skipping"
            )
            skipped.append(dev)
            continue

        if not test_records and not atk_records:
            logger.warning(f"  {dev}: no test or attack windows — skipping")
            skipped.append(dev)
            continue

        # Scale using the global scaler (same scaler NEUROGUARD uses)
        X_train = scaler.transform(
            np.stack([r.features for r in train_records])
        ).astype(np.float32)

        X_test_n = (
            scaler.transform(np.stack([r.features for r in test_records])).astype(np.float32)
            if test_records else np.empty((0, 60), dtype=np.float32)
        )
        X_attack = (
            scaler.transform(np.stack([r.features for r in atk_records])).astype(np.float32)
            if atk_records else np.empty((0, 60), dtype=np.float32)
        )

        logger.info(
            f"  {dev}  train={len(X_train)}  "
            f"test_n={len(X_test_n)}  attack={len(X_attack)}"
        )

        # ── IsolationForest ──────────────────────────────────────────────────
        iforest = IsolationForest(
            n_estimators=200,
            contamination="auto",   # auto = no assumed contamination
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        iforest.fit(X_train)

        if len(X_test_n):
            all_scores_if_normal.extend((-iforest.score_samples(X_test_n)).tolist())
        if len(X_attack):
            all_scores_if_attack.extend((-iforest.score_samples(X_attack)).tolist())

        # ── OneClassSVM ───────────────────────────────────────────────────────
        ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        ocsvm.fit(X_train)

        if len(X_test_n):
            all_scores_svm_normal.extend((-ocsvm.decision_function(X_test_n)).tolist())
        if len(X_attack):
            all_scores_svm_attack.extend((-ocsvm.decision_function(X_attack)).tolist())

        # ── Autoencoder ───────────────────────────────────────────────────────
        ae = _train_autoencoder(X_train, ae_device)

        if len(X_test_n):
            all_scores_ae_normal.extend(_ae_recon_errors(ae, X_test_n, ae_device).tolist())
        if len(X_attack):
            all_scores_ae_attack.extend(_ae_recon_errors(ae, X_attack, ae_device).tolist())

    # ── 4. Aggregate metrics ──────────────────────────────────────────────────
    logger.info("Computing aggregate metrics across all devices…")

    results = {}

    for name, s_normal, s_attack in [
        ("IsolationForest (per-device)", all_scores_if_normal,  all_scores_if_attack),
        ("OneClassSVM (per-device)",     all_scores_svm_normal, all_scores_svm_attack),
        ("Autoencoder (per-device)",     all_scores_ae_normal,  all_scores_ae_attack),
    ]:
        if not s_normal or not s_attack:
            logger.warning(f"{name}: insufficient scores — skipping")
            continue

        sn = np.array(s_normal)
        sa = np.array(s_attack)

        labels_all = [0] * len(sn) + [1] * len(sa)
        scores_all = sn.tolist() + sa.tolist()

        roc = roc_auc_score(labels_all, scores_all)

        # Threshold: p95 of normal scores (mirrors NEUROGUARD's k=2.5 operating point)
        threshold_p95 = float(np.percentile(sn, 95))
        conf = _confusion(s_normal, s_attack, threshold_p95)

        results[name] = {
            "roc_auc":   roc,
            "tpr":       conf["tpr"],
            "fpr":       conf["fpr"],
            "f1":        conf["f1"],
            "precision": conf["precision"],
            "tp": conf["tp"], "fp": conf["fp"],
            "fn": conf["fn"], "tn": conf["tn"],
            "n_normal":  len(sn),
            "n_attack":  len(sa),
            "threshold": threshold_p95,
        }

        logger.info(
            f"{name}: ROC-AUC={roc:.4f}  TPR={conf['tpr']:.4f}  "
            f"FPR={conf['fpr']:.4f}  F1={conf['f1']:.4f}"
        )

    # ── 5. Save results ───────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved → {OUT_PATH}")

    # ── 6. Print comparison table ─────────────────────────────────────────────
    W = 82
    print(f"\n{'═'*W}")
    print(f"  NEUROGUARD — FAIR PER-DEVICE BASELINE COMPARISON")
    print(f"  All baselines trained per-device (same structural treatment as NEUROGUARD)")
    print(f"{'═'*W}")
    print(
        f"  {'Method':<34}  {'ROC-AUC':>8}  {'TPR':>7}  {'FPR':>7}  "
        f"{'F1':>7}  {'TP':>5}  {'FP':>5}"
    )
    print(f"  {'─'*34}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}")

    for name, m in results.items():
        print(
            f"  {name:<34}  {m['roc_auc']:>8.4f}  "
            f"{m['tpr']:>6.1%}  {m['fpr']:>6.1%}  "
            f"{m['f1']:>7.4f}  {m['tp']:>5}  {m['fp']:>5}"
        )

    print(f"  {'─'*34}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}")

    ng = NEUROGUARD
    print(
        f"  {'NEUROGUARD (k=2.5)  [ours]':<34}  {ng['roc_auc']:>8.4f}  "
        f"{ng['tpr_25']:>6.1%}  {ng['fpr_25']:>6.1%}  "
        f"{ng['f1_25']:>7.4f}  {'—':>5}  {'—':>5}"
    )
    print(
        f"  {'NEUROGUARD (k=3.0)  [ours]':<34}  {ng['roc_auc']:>8.4f}  "
        f"{ng['tpr_30']:>6.1%}  {ng['fpr_30']:>6.1%}  "
        f"{ng['f1_30']:>7.4f}  {'—':>5}  {'—':>5}"
    )
    print(f"{'═'*W}")

    if skipped:
        print(f"\n  Skipped devices (insufficient data): {', '.join(skipped)}")

    print(f"\n  Note: Threshold set at p95 of per-device normal scores.")
    print(f"  This mirrors NEUROGUARD's k=2.5 sigma operating point.\n")


if __name__ == "__main__":
    main()
