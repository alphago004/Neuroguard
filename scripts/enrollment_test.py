"""
NEUROGUARD — Enrollment + scoring smoke test.

Enrolls all 16 devices from test_normal windows, then scores:
  - 20 randomly sampled NORMAL windows (from test_normal holdout)
  - 20 randomly sampled ATTACK windows (from attack records)

Reports anomaly scores side-by-side and summarizes detection rate.
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from src.training.dataset import WindowDataset, WindowRecord, LABEL_NORMAL, LABEL_ATTACK
from src.detection.enroll import enroll_all_devices, DeviceDNA
from src.detection.scorer import score_records, score_window, STATUS_ALERT

CHECKPOINT   = ROOT / "models" / "checkpoints" / "best_model.pt"
SCALER_PATH  = ROOT / "models" / "checkpoints" / "scaler.pkl"
CACHE_PATH   = ROOT / "data" / "processed" / "window_dataset.pkl"

N_SAMPLE = 20
SEED     = 42

# ── Load dataset + scaler ──────────────────────────────────────────────────
logger.info("Loading WindowDataset and scaler…")
window_ds = WindowDataset.load(CACHE_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ── Enroll all devices ─────────────────────────────────────────────────────
logger.info("Enrolling all devices from test_normal windows…")
dna_map = enroll_all_devices(window_ds, scaler, checkpoint_path=CHECKPOINT)

print(f"\n{'─'*60}")
print(f"  ENROLLED {len(dna_map)} DEVICES")
print(f"{'─'*60}")
print(f"  {'Device IP':<20} {'Windows':>8}  {'Threshold':>10}")
print(f"  {'─'*18:<20} {'─'*8:>8}  {'─'*10:>10}")
for dev_id, dna in sorted(dna_map.items()):
    print(f"  {dev_id:<20} {dna.n_windows:>8}  {dna.threshold_distance:>10.4f}")

# ── Scale all test/attack windows ──────────────────────────────────────────
def scale_records(records, scaler):
    if not records:
        return []
    X = np.stack([r.features for r in records])
    X_s = scaler.transform(X).astype(np.float32)
    return [
        WindowRecord(r.device_id, X_s[i], r.label, r.window_idx, r.flow_start)
        for i, r in enumerate(records)
    ]

scaled_test   = scale_records(window_ds.test_normal, scaler)
scaled_attack = scale_records(window_ds.attack_records, scaler)

# Only keep attack windows whose device has enrolled DNA
enrolled_ids = set(dna_map.keys())
attack_eligible = [r for r in scaled_attack if r.device_id in enrolled_ids]
normal_eligible = [r for r in scaled_test   if r.device_id in enrolled_ids]

rng = random.Random(SEED)
sample_normal = rng.sample(normal_eligible, min(N_SAMPLE, len(normal_eligible)))
sample_attack = rng.sample(attack_eligible, min(N_SAMPLE, len(attack_eligible)))

# ── Score both groups ──────────────────────────────────────────────────────
logger.info(f"Scoring {len(sample_normal)} normal and {len(sample_attack)} attack windows…")

results_normal = score_records(sample_normal, dna_map, checkpoint_path=CHECKPOINT)
results_attack = score_records(sample_attack, dna_map, checkpoint_path=CHECKPOINT, compute_attribution=True)

# ── Side-by-side report ────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  ANOMALY SCORES — NORMAL WINDOWS (should be < 1.0)")
print(f"{'═'*70}")
print(f"  {'#':<4} {'Device':<20} {'Score':>8}  {'Raw Dist':>10}  {'Threshold':>10}  Status")
print(f"  {'─'*4:<4} {'─'*20:<20} {'─'*8:>8}  {'─'*10:>10}  {'─'*10:>10}  {'─'*6}")
for i, r in enumerate(results_normal, 1):
    flag = "✓" if r.status == "NORMAL" else "✗ FP"
    print(f"  {i:<4} {r.device_id:<20} {r.anomaly_score:>8.4f}  {r.raw_distance:>10.4f}  {r.threshold:>10.4f}  {flag}")

normal_scores = [r.anomaly_score for r in results_normal]
n_fp = sum(1 for r in results_normal if r.status == STATUS_ALERT)
print(f"\n  Normal  → mean={np.mean(normal_scores):.4f}  std={np.std(normal_scores):.4f}  "
      f"max={np.max(normal_scores):.4f}  FP={n_fp}/{len(results_normal)}")

print(f"\n{'═'*70}")
print(f"  ANOMALY SCORES — ATTACK WINDOWS (should be ≥ 1.0)")
print(f"{'═'*70}")
print(f"  {'#':<4} {'Device':<20} {'Score':>8}  {'Raw Dist':>10}  {'Threshold':>10}  Status  Top Feature")
print(f"  {'─'*4:<4} {'─'*20:<20} {'─'*8:>8}  {'─'*10:>10}  {'─'*10:>10}  {'─'*6}  {'─'*16}")
for i, r in enumerate(results_attack, 1):
    flag  = "ALERT ✓" if r.status == STATUS_ALERT else "missed"
    feat  = r.top_features[0] if r.top_features else "—"
    print(f"  {i:<4} {r.device_id:<20} {r.anomaly_score:>8.4f}  {r.raw_distance:>10.4f}  {r.threshold:>10.4f}  {flag:<7}  {feat}")

attack_scores = [r.anomaly_score for r in results_attack]
n_detected = sum(1 for r in results_attack if r.status == STATUS_ALERT)
print(f"\n  Attack  → mean={np.mean(attack_scores):.4f}  std={np.std(attack_scores):.4f}  "
      f"min={np.min(attack_scores):.4f}  Detected={n_detected}/{len(results_attack)}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  DETECTION SUMMARY")
print(f"{'═'*70}")
print(f"  True Positives  (attack detected):  {n_detected:>3} / {len(results_attack)}"
      f"  ({n_detected/len(results_attack)*100:.1f}%)")
print(f"  False Positives (normal flagged):   {n_fp:>3} / {len(results_normal)}"
      f"  ({n_fp/len(results_normal)*100:.1f}%)")
gap = np.mean(attack_scores) - np.mean(normal_scores)
print(f"  Score gap (attack_mean - normal_mean): {gap:+.4f}")
print(f"{'═'*70}\n")
