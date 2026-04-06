"""NEUROGUARD — Full paper evaluation runner."""

import pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from src.training.dataset import WindowDataset
from src.detection.enroll import enroll_all_devices
from src.training.metrics import evaluate_model, plot_roc_curve

CHECKPOINT  = ROOT / "models" / "checkpoints" / "best_model.pt"
SCALER_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"
CACHE_PATH  = ROOT / "data" / "processed" / "window_dataset.pkl"
ROC_PATH    = ROOT / "models" / "checkpoints" / "roc_curve.png"

logger.info("Loading dataset and scaler…")
window_ds = WindowDataset.load(CACHE_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

logger.info("Enrolling all devices…")
dna_map = enroll_all_devices(window_ds, scaler, checkpoint_path=CHECKPOINT)

logger.info("Running full evaluation…")
report = evaluate_model(window_ds, dna_map, scaler, checkpoint_path=CHECKPOINT)

plot_roc_curve(report, output_path=ROC_PATH)

# ── Print full report ──────────────────────────────────────────────────────
W = 68

print(f"\n{'═'*W}")
print(f"  NEUROGUARD — FULL EVALUATION REPORT (TON_IoT)")
print(f"{'═'*W}")
print(f"  Normal windows scored : {report.n_normal:>6}")
print(f"  Attack windows scored : {report.n_attack:>6}")
print(f"  Enrolled devices      : {len(report.enrolled_devices):>6}")
print(f"\n  ── PRIMARY METRIC ────────────────────────────────────────────")
print(f"  ROC-AUC               : {report.roc_auc:.4f}   (target > 0.95)")

# ── Confusion matrices ──────────────────────────────────────────────────
for conf, label in [(report.confusion_2_5, "k=2.5 (default)"),
                    (report.confusion_3_0, "k=3.0 (relaxed)")]:
    print(f"\n  ── CONFUSION MATRIX  {label} {'─'*(W-24-len(label))}")
    print(f"  {'':20}  Predicted NORMAL  Predicted ALERT")
    print(f"  {'Actual NORMAL':20}  {conf.tn:>16}  {conf.fp:>15}")
    print(f"  {'Actual ATTACK':20}  {conf.fn:>16}  {conf.tp:>15}")
    print(f"")
    print(f"  Detection Rate (TPR) : {conf.tpr:.4f}  ({conf.tp}/{conf.tp+conf.fn})")
    print(f"  False Positive Rate  : {conf.fpr:.4f}  ({conf.fp}/{conf.fp+conf.tn})")
    print(f"  Precision            : {conf.precision:.4f}")
    print(f"  F1 Score             : {conf.f1:.4f}")
    print(f"  Accuracy             : {conf.accuracy:.4f}")

# ── Per-attack-type breakdown ──────────────────────────────────────────
print(f"\n  ── PER-ATTACK-TYPE DETECTION (k=2.5) {'─'*(W-38)}")
print(f"  {'Attack Type':<14} {'Windows':>8}  {'Detected':>9}  {'Det. Rate':>10}  "
      f"{'Mean Score':>11}  {'Min':>8}  {'Max':>8}")
print(f"  {'─'*14:<14} {'─'*8:>8}  {'─'*9:>9}  {'─'*10:>10}  "
      f"{'─'*11:>11}  {'─'*8:>8}  {'─'*8:>8}")

# Sort by detection rate descending
for atype, m in sorted(report.per_type.items(), key=lambda x: -x[1].detection_rate):
    bar = "█" * int(m.detection_rate * 20)
    tgt = " ✓" if m.detection_rate >= 0.90 else " ✗"
    print(f"  {atype:<14} {m.n_windows:>8}  {m.n_detected:>9}  "
          f"{m.detection_rate:>9.1%}  {m.mean_score:>11.4f}  "
          f"{m.min_score:>8.4f}  {m.max_score:>8.4f}{tgt}")

# ── Score distribution summary ────────────────────────────────────────
sn = np.array(report.scores_normal)
sa = np.array(report.scores_attack)
print(f"\n  ── SCORE DISTRIBUTIONS {'─'*(W-23)}")
print(f"  {'':12}  {'Mean':>8}  {'Std':>8}  {'p5':>8}  {'p25':>8}  "
      f"{'Median':>8}  {'p75':>8}  {'p95':>8}")
print(f"  {'Normal':12}  {sn.mean():>8.4f}  {sn.std():>8.4f}  "
      f"{np.percentile(sn,5):>8.4f}  {np.percentile(sn,25):>8.4f}  "
      f"{np.median(sn):>8.4f}  {np.percentile(sn,75):>8.4f}  "
      f"{np.percentile(sn,95):>8.4f}")
print(f"  {'Attack':12}  {sa.mean():>8.4f}  {sa.std():>8.4f}  "
      f"{np.percentile(sa,5):>8.4f}  {np.percentile(sa,25):>8.4f}  "
      f"{np.median(sa):>8.4f}  {np.percentile(sa,75):>8.4f}  "
      f"{np.percentile(sa,95):>8.4f}")
print(f"  Score gap (attack p5 - normal p95): "
      f"{np.percentile(sa,5) - np.percentile(sn,95):+.4f}")

print(f"\n  ROC curve saved → {ROC_PATH}")
print(f"{'═'*W}\n")
