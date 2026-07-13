"""
Multi-seed evaluation (Step 2 — M6 fix).

Runs the full NEUROGUARD pipeline 5 times with different random seeds.
Each seed controls the 80/14/6 per-device split and the model weight
initialization, demonstrating that results are not an artifact of a lucky
data partition.

Usage:
    python3 scripts/multi_seed_eval.py

Output:
    multi_seed_results.txt  (in project root)

Seeds used: 42 (canonical), 1, 2, 3, 4
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from loguru import logger

from src.training.dataset import WindowDataset, PairDataset, WindowRecord, build_windows
from src.models.siamese import SiameseNetwork, build_model
from src.detection.enroll import enroll_all_devices
from src.training.metrics import evaluate_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TON_IOT_CSV  = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
WINDOW_CACHE = PROJECT_ROOT / "data" / "processed" / "window_dataset.pkl"
SEED_DIR     = PROJECT_ROOT / "models" / "checkpoints" / "multi_seed"
RESULTS_PATH = PROJECT_ROOT / "multi_seed_results.txt"

# ---------------------------------------------------------------------------
# Hyperparameters — identical to main training
# ---------------------------------------------------------------------------
SEEDS      = [42, 1, 2, 3, 4]
EPOCHS     = 100
BATCH_SIZE = 128
LR         = 1e-3
PATIENCE   = 100
DEVICE_CAP = 50
N_PAIRS    = 100_000
MARGIN     = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_scaler(records: list[WindowRecord]) -> RobustScaler:
    X = np.stack([r.features for r in records])
    scaler = RobustScaler()
    scaler.fit(X)
    return scaler


def apply_scaler(records: list[WindowRecord], scaler: RobustScaler) -> list[WindowRecord]:
    if not records:
        return []
    X = np.stack([r.features for r in records])
    X_scaled = scaler.transform(X).astype(np.float32)
    return [
        WindowRecord(
            device_id=r.device_id, features=X_scaled[i],
            label=r.label, window_idx=r.window_idx, flow_start=r.flow_start,
        )
        for i, r in enumerate(records)
    ]


def run_one_seed(seed: int, raw_records: list[WindowRecord], device: torch.device) -> dict:
    """Train and evaluate one full run with the given split seed."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SEED {seed}")
    logger.info(f"{'=' * 60}")

    t_start = time.time()

    # ── Build dataset split ────────────────────────────────────────────────
    ds = WindowDataset(raw_records, seed=seed)
    logger.info(
        f"Split (seed={seed}): train={len(ds.train_normal)} "
        f"enroll={len(ds.enroll_normal)} test={len(ds.test_normal)} "
        f"attack={len(ds.attack_records)}"
    )

    # ── Scaler (fit on THIS seed's train_normal) ──────────────────────────
    scaler = fit_scaler(ds.train_normal)

    # ── Scale all splits ──────────────────────────────────────────────────
    scaled_train  = apply_scaler(ds.train_normal,  scaler)
    scaled_enroll = apply_scaler(ds.enroll_normal, scaler)
    scaled_test   = apply_scaler(ds.test_normal,   scaler)

    # Carve internal val from last 10% of train_normal per device (chronological)
    per_dev = {}
    for r in scaled_train:
        per_dev.setdefault(r.device_id, []).append(r)
    actual_train: list[WindowRecord] = []
    internal_val: list[WindowRecord] = []
    for recs in per_dev.values():
        recs_s = sorted(recs, key=lambda r: r.window_idx)
        n_val = max(1, int(len(recs_s) * 0.10))
        actual_train.extend(recs_s[:-n_val])
        internal_val.extend(recs_s[-n_val:])

    scaled_ds = WindowDataset.__new__(WindowDataset)
    scaled_ds.records        = actual_train + internal_val + scaled_enroll + scaled_test
    scaled_ds.train_normal   = actual_train
    scaled_ds.enroll_normal  = internal_val   # internal val for early stopping
    scaled_ds.test_normal    = scaled_test
    scaled_ds.attack_records = apply_scaler(ds.attack_records, scaler)
    scaled_ds.device_ids     = ds.device_ids
    scaled_ds.device_to_idx  = ds.device_to_idx
    scaled_ds._seed          = seed

    # ── Pair datasets ─────────────────────────────────────────────────────
    val_pairs  = PairDataset(scaled_ds, split="enroll", n_pairs=N_PAIRS,
                              device_cap=DEVICE_CAP, seed=0)
    val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model, loss_fn = build_model(transformer_layers=1, margin=MARGIN)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    checkpoint_path = SEED_DIR / f"model_seed{seed}.pt"

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improv = 0

    logger.info(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Status'}")
    logger.info("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        epoch_pairs = PairDataset(
            scaled_ds, split="train",
            n_pairs=N_PAIRS, device_cap=DEVICE_CAP, seed=epoch * 7 + seed,
        )
        train_loader = DataLoader(epoch_pairs, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0)

        model.train()
        running_loss = 0.0
        for anchor, pair, labels in train_loader:
            anchor, pair, labels = anchor.to(device), pair.to(device), labels.to(device)
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
                anchor, pair, labels = anchor.to(device), pair.to(device), labels.to(device)
                emb_a, emb_b = model(anchor, pair)
                dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
                val_running += loss_fn(dist, labels).item()
        val_loss = val_running / len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improv = 0
            torch.save(model.state_dict(), checkpoint_path)
            status = "saved"
        else:
            epochs_no_improv += 1
            status = f"no improv {epochs_no_improv}/{PATIENCE}"

        if epoch % 10 == 0 or status == "saved":
            logger.info(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  {status}")

        if epochs_no_improv >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Seed {seed}: best epoch={best_epoch}, val_loss={best_val_loss:.6f}")

    # ── Evaluation ────────────────────────────────────────────────────────
    dna_map = enroll_all_devices(ds, scaler, checkpoint_path=checkpoint_path)
    report  = evaluate_model(ds, dna_map, scaler, checkpoint_path=checkpoint_path)

    elapsed = time.time() - t_start
    logger.info(
        f"Seed {seed}: AUC={report.roc_auc:.4f} "
        f"TPR={report.confusion_2_5.tpr:.4f} "
        f"FPR={report.confusion_2_5.fpr:.4f} "
        f"F1={report.confusion_2_5.f1:.4f} "
        f"({elapsed/60:.1f}min)"
    )

    return {
        "seed":       seed,
        "roc_auc":    report.roc_auc,
        "tpr":        report.confusion_2_5.tpr,
        "fpr":        report.confusion_2_5.fpr,
        "f1":         report.confusion_2_5.f1,
        "best_epoch": best_epoch,
        "elapsed_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger.info(f"Multi-seed eval: device={device}, seeds={SEEDS}")

    # Load raw window records once (window extraction is deterministic)
    if WINDOW_CACHE.exists():
        logger.info(f"Loading window cache: {WINDOW_CACHE}")
        cached_ds = WindowDataset.load(WINDOW_CACHE)
        raw_records = cached_ds.records
    else:
        logger.info("Building window dataset from CSV…")
        cached_ds = build_windows(TON_IOT_CSV)
        raw_records = cached_ds.records

    logger.info(f"Raw records: {len(raw_records):,} total windows")

    results = []
    for seed in SEEDS:
        res = run_one_seed(seed, raw_records, device)
        results.append(res)

    # ── Aggregate ────────────────────────────────────────────────────────
    auc_vals = [r["roc_auc"] for r in results]
    tpr_vals = [r["tpr"]     for r in results]
    fpr_vals = [r["fpr"]     for r in results]
    f1_vals  = [r["f1"]      for r in results]

    auc_mean, auc_std = float(np.mean(auc_vals)), float(np.std(auc_vals))
    tpr_mean, tpr_std = float(np.mean(tpr_vals)), float(np.std(tpr_vals))
    fpr_mean, fpr_std = float(np.mean(fpr_vals)), float(np.std(fpr_vals))
    f1_mean,  f1_std  = float(np.mean(f1_vals)),  float(np.std(f1_vals))

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ROC-AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
    logger.info(f"  TPR:      {tpr_mean:.4f} ± {tpr_std:.4f}")
    logger.info(f"  FPR:      {fpr_mean:.4f} ± {fpr_std:.4f}")
    logger.info(f"  F1:       {f1_mean:.4f}  ± {f1_std:.4f}")
    logger.info("=" * 60)
    logger.info("Per-seed breakdown:")
    for r in results:
        logger.info(
            f"  seed={r['seed']:2d}  AUC={r['roc_auc']:.4f}  "
            f"TPR={r['tpr']:.4f}  FPR={r['fpr']:.4f}  F1={r['f1']:.4f}"
        )

    with open(RESULTS_PATH, "w") as f:
        f.write("MULTI-SEED EVALUATION RESULTS\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("Per-seed results:\n")
        for r in results:
            f.write(
                f"  seed={r['seed']:2d}  AUC={r['roc_auc']:.4f}  "
                f"TPR={r['tpr']:.4f}  FPR={r['fpr']:.4f}  F1={r['f1']:.4f}  "
                f"best_epoch={r['best_epoch']}  elapsed={r['elapsed_s']:.0f}s\n"
            )
        f.write(f"\nMean ± Std (n={len(SEEDS)}):\n")
        f.write(f"  ROC-AUC:  {auc_mean:.4f} ± {auc_std:.4f}\n")
        f.write(f"  TPR:      {tpr_mean:.4f} ± {tpr_std:.4f}\n")
        f.write(f"  FPR:      {fpr_mean:.4f} ± {fpr_std:.4f}\n")
        f.write(f"  F1:       {f1_mean:.4f}  ± {f1_std:.4f}\n")

    logger.info(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
