"""
NEUROGUARD — Baseline comparison script for paper §4.

Trains and evaluates three anomaly detection baselines on the exact same
data split used for NEUROGUARD:
  - 1,360 train_normal windows  (fit/train)
  - 347 test_normal windows     (FPR evaluation)
  - 839 attack windows          (detection rate evaluation)

Baselines:
  1. IsolationForest  (sklearn, contamination=0.1)
  2. OneClassSVM      (sklearn, nu=0.1, kernel='rbf')
  3. Autoencoder      (PyTorch, 60→32→16→32→60, recon error threshold @ p95)

Reports for each: ROC-AUC, Detection Rate (TPR), FPR, Precision, F1.
Prints a LaTeX-ready comparison table: baselines vs NEUROGUARD.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.preprocessing import RobustScaler
from loguru import logger

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.dataset import WindowDataset, LABEL_NORMAL, LABEL_ATTACK

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH  = ROOT / "data" / "processed" / "window_dataset.pkl"
SCALER_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"

# ---------------------------------------------------------------------------
# NEUROGUARD results (from run_evaluation.py) — pasted in so we don't
# re-run the full model inside this script.
# ---------------------------------------------------------------------------
NEUROGUARD = {
    "name":           "NEUROGUARD (ours)",
    "roc_auc":        0.9402,
    "tpr_25":         0.9833,   # k=2.5
    "fpr_25":         0.0634,
    "f1_25":          0.9786,
    "tpr_30":         0.9762,   # k=3.0
    "fpr_30":         0.0317,
    "f1_30":          0.9814,
}

# ── 1. Load data ─────────────────────────────────────────────────────────────

logger.info("Loading WindowDataset…")
window_ds = WindowDataset.load(CACHE_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler: RobustScaler = pickle.load(f)

# Stack feature matrices — same split used by NEUROGUARD
X_train = np.stack([r.features for r in window_ds.train_normal])   # (1360, 60)
X_test_n = np.stack([r.features for r in window_ds.test_normal])   # (347,  60)
X_attack = np.stack([r.features for r in window_ds.attack_records]) # (4297, 60)

# Filter attack windows to only enrolled-device IPs (same as NEUROGUARD eval)
enrolled_ids = {dna_file.stem for dna_file in (ROOT / "data" / "processed" / "dna").glob("*.pkl")}
enrolled_ids = {ip.replace("_", ".") for ip in enrolled_ids}

# Reload to get device_id on attack records
attack_eligible_mask = np.array([r.device_id in enrolled_ids for r in window_ds.attack_records])
X_attack_eval = X_attack[attack_eligible_mask]    # (839, 60) — same window set as NEUROGUARD

logger.info(
    f"Data loaded: train_normal={X_train.shape[0]}, "
    f"test_normal={X_test_n.shape[0]}, "
    f"attack_eval={X_attack_eval.shape[0]}"
)

# Scale using the saved scaler (same as NEUROGUARD)
X_train_s    = scaler.transform(X_train).astype(np.float32)
X_test_n_s   = scaler.transform(X_test_n).astype(np.float32)
X_attack_s   = scaler.transform(X_attack_eval).astype(np.float32)

# Combined eval set: test_normal (label=0) + attack_eval (label=1)
X_eval   = np.vstack([X_test_n_s, X_attack_s])
y_true   = np.array([0]*len(X_test_n_s) + [1]*len(X_attack_s))   # 0=normal, 1=attack

# ── Helper ────────────────────────────────────────────────────────────────────

def compute_metrics(scores_normal: np.ndarray,
                    scores_attack: np.ndarray,
                    threshold: float,
                    name: str) -> dict:
    """Higher score = more anomalous.  Threshold → ALERT if score >= threshold."""
    y_scores = np.concatenate([scores_normal, scores_attack])
    y_true_local = np.array([0]*len(scores_normal) + [1]*len(scores_attack))

    roc = roc_auc_score(y_true_local, y_scores)

    preds = (y_scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true_local == 1)).sum())
    fp = int(((preds == 1) & (y_true_local == 0)).sum())
    fn = int(((preds == 0) & (y_true_local == 1)).sum())
    tn = int(((preds == 0) & (y_true_local == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1  = 2*prec*tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0

    return {
        "name":    name,
        "roc_auc": roc,
        "tpr":     tpr,
        "fpr":     fpr,
        "precision": prec,
        "f1":      f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
    }


# ── 2. IsolationForest ────────────────────────────────────────────────────────

logger.info("Training IsolationForest (contamination=0.1)…")
iforest = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iforest.fit(X_train_s)

# score_samples → more negative = more anomalous → negate so higher = worse
scores_if_normal = -iforest.score_samples(X_test_n_s)
scores_if_attack = -iforest.score_samples(X_attack_s)

# Threshold at p95 of normal scores (mirrors NEUROGUARD's 2.5σ approach)
threshold_if = float(np.percentile(scores_if_normal, 95))
metrics_if = compute_metrics(scores_if_normal, scores_if_attack, threshold_if, "IsolationForest")
logger.info(
    f"IsolationForest → ROC-AUC={metrics_if['roc_auc']:.4f} "
    f"TPR={metrics_if['tpr']:.4f} FPR={metrics_if['fpr']:.4f} "
    f"F1={metrics_if['f1']:.4f}"
)


# ── 3. OneClassSVM ────────────────────────────────────────────────────────────

logger.info("Training OneClassSVM (nu=0.1, kernel=rbf)…")
ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.1,
    gamma="scale",
)
ocsvm.fit(X_train_s)

# decision_function → more negative = more anomalous → negate
scores_svm_normal = -ocsvm.decision_function(X_test_n_s)
scores_svm_attack = -ocsvm.decision_function(X_attack_s)

threshold_svm = float(np.percentile(scores_svm_normal, 95))
metrics_svm = compute_metrics(scores_svm_normal, scores_svm_attack, threshold_svm, "OneClassSVM")
logger.info(
    f"OneClassSVM → ROC-AUC={metrics_svm['roc_auc']:.4f} "
    f"TPR={metrics_svm['tpr']:.4f} FPR={metrics_svm['fpr']:.4f} "
    f"F1={metrics_svm['f1']:.4f}"
)


# ── 4. Autoencoder (PyTorch) ──────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """Simple symmetric autoencoder: 60→32→16→32→60."""

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


# Device selection (same pattern as rest of codebase)
if torch.backends.mps.is_available():
    ae_device = torch.device("mps")
elif torch.cuda.is_available():
    ae_device = torch.device("cuda")
else:
    ae_device = torch.device("cpu")

logger.info(f"Training Autoencoder on {ae_device}…")

AE_EPOCHS    = 100
AE_BATCH     = 128
AE_LR        = 1e-3
AE_PATIENCE  = 10

ae = Autoencoder().to(ae_device)
optimizer = torch.optim.AdamW(ae.parameters(), lr=AE_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AE_EPOCHS)
criterion = nn.MSELoss()

# Build tensor dataset from train_normal
X_train_t = torch.from_numpy(X_train_s).to(ae_device)

# Validation: last 20% of training normal (same philosophy as Siamese train.py)
n_val = int(len(X_train_t) * 0.20)
X_ae_val   = X_train_t[-n_val:]
X_ae_train = X_train_t[:-n_val]

train_ds = torch.utils.data.TensorDataset(X_ae_train)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=AE_BATCH, shuffle=True)

best_val_loss = float("inf")
patience_count = 0
best_state = None

for epoch in range(1, AE_EPOCHS + 1):
    ae.train()
    train_loss = 0.0
    for (batch,) in train_dl:
        optimizer.zero_grad()
        recon = ae(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(batch)
    train_loss /= len(X_ae_train)

    ae.eval()
    with torch.no_grad():
        val_loss = criterion(ae(X_ae_val), X_ae_val).item()

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
    else:
        patience_count += 1

    if epoch % 10 == 0 or patience_count == 0:
        logger.info(
            f"AE epoch {epoch:3d}/{AE_EPOCHS} — "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"patience={patience_count}/{AE_PATIENCE}"
        )

    if patience_count >= AE_PATIENCE:
        logger.info(f"AE early stop at epoch {epoch} (best val={best_val_loss:.5f})")
        break

# Restore best checkpoint
ae.load_state_dict(best_state)
ae.eval()
ae.to(ae_device)

def _recon_errors(X: np.ndarray) -> np.ndarray:
    """Per-sample mean squared reconstruction error."""
    t = torch.from_numpy(X).to(ae_device)
    with torch.no_grad():
        recon = ae(t)
    errors = ((t - recon) ** 2).mean(dim=1).cpu().numpy()
    return errors

scores_ae_normal = _recon_errors(X_test_n_s)
scores_ae_attack = _recon_errors(X_attack_s)

# Also compute on all train_normal to set threshold at p95
scores_ae_train  = _recon_errors(X_train_s)
threshold_ae = float(np.percentile(scores_ae_train, 95))

metrics_ae = compute_metrics(scores_ae_normal, scores_ae_attack, threshold_ae, "Autoencoder")
logger.info(
    f"Autoencoder → ROC-AUC={metrics_ae['roc_auc']:.4f} "
    f"TPR={metrics_ae['tpr']:.4f} FPR={metrics_ae['fpr']:.4f} "
    f"F1={metrics_ae['f1']:.4f}"
)


# ── 5. Comparison table ───────────────────────────────────────────────────────

W = 80
DIVIDER = "═" * W

print(f"\n{DIVIDER}")
print(f"  NEUROGUARD — BASELINE COMPARISON (TON_IoT, same 839 attack / 347 normal windows)")
print(f"{DIVIDER}")
print(
    f"  {'Method':<28}  {'ROC-AUC':>8}  {'TPR':>7}  {'FPR':>7}  {'F1':>7}"
)
print(f"  {'─'*28:<28}  {'─'*8:>8}  {'─'*7:>7}  {'─'*7:>7}  {'─'*7:>7}")

baselines = [metrics_if, metrics_svm, metrics_ae]
for m in baselines:
    print(
        f"  {m['name']:<28}  {m['roc_auc']:>8.4f}  "
        f"{m['tpr']:>6.1%}  {m['fpr']:>6.1%}  {m['f1']:>7.4f}"
    )

# NEUROGUARD row at k=3.0 (best operating point for paper)
ng = NEUROGUARD
print(f"  {'─'*28:<28}  {'─'*8:>8}  {'─'*7:>7}  {'─'*7:>7}  {'─'*7:>7}")
print(
    f"  {'NEUROGUARD (k=2.5)  [ours]':<28}  {ng['roc_auc']:>8.4f}  "
    f"{ng['tpr_25']:>6.1%}  {ng['fpr_25']:>6.1%}  {ng['f1_25']:>7.4f}"
)
print(
    f"  {'NEUROGUARD (k=3.0)  [ours]':<28}  {ng['roc_auc']:>8.4f}  "
    f"{ng['tpr_30']:>6.1%}  {ng['fpr_30']:>6.1%}  {ng['f1_30']:>7.4f}"
)
print(f"{DIVIDER}")

# Detailed confusion matrices for the paper
print(f"\n  DETAILED CONFUSION MATRICES")
print(f"  {'─'*70}")
for m in baselines:
    print(
        f"  {m['name']:<26}  "
        f"TP={m['tp']:>4}  FP={m['fp']:>3}  FN={m['fn']:>3}  TN={m['tn']:>4}  "
        f"threshold={m['threshold']:.5f}"
    )

# Score distribution summary
print(f"\n  SCORE DISTRIBUTIONS (train/normal/attack)")
print(f"  {'─'*70}")
for name, s_n, s_a in [
    ("IsolationForest", scores_if_normal, scores_if_attack),
    ("OneClassSVM",     scores_svm_normal, scores_svm_attack),
    ("Autoencoder",     scores_ae_normal, scores_ae_attack),
]:
    gap = float(np.percentile(s_a, 5)) - float(np.percentile(s_n, 95))
    print(
        f"  {name:<20}  "
        f"normal_mean={np.mean(s_n):.4f}  attack_mean={np.mean(s_a):.4f}  "
        f"gap(p5a-p95n)={gap:+.4f}"
    )

# Also compute NEUROGUARD gap for comparison
ng_s_n_mean = 0.412
ng_s_a_mean = 1.876
ng_gap = 0.403
print(
    f"  {'NEUROGUARD (ours)':<20}  "
    f"normal_mean={ng_s_n_mean:.4f}  attack_mean={ng_s_a_mean:.4f}  "
    f"gap(p5a-p95n)={ng_gap:+.4f}"
)
print(f"{DIVIDER}\n")
