"""
End-to-end NEUROGUARD pipeline: train → enroll → evaluate.

Runs the full pipeline from raw CSV to final evaluation report in a single
command. Useful for reproducing paper results or testing configuration changes.

Usage:
    python scripts/train_and_evaluate.py
    python scripts/train_and_evaluate.py --epochs 50 --patience 20
    python scripts/train_and_evaluate.py --no-cache  # ignore window cache
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.training.train import train
from src.detection.enroll import enroll_all_devices
from src.training.dataset import WindowDataset
from src.training.metrics import evaluate_model, plot_roc_curve

import pickle

TON_IOT_CSV    = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
WINDOW_CACHE   = PROJECT_ROOT / "data" / "processed" / "window_dataset.pkl"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
SCALER_PKL     = CHECKPOINT_DIR / "scaler.pkl"
BEST_MODEL_PT  = CHECKPOINT_DIR / "best_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NEUROGUARD and evaluate on TON_IoT"
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=100)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Rebuild window dataset from scratch")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip ROC curve plot")
    args = parser.parse_args()

    t_total = time.time()

    # ── Step 1: Train ─────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 — TRAINING")
    logger.info("=" * 55)
    train_results = train(
        csv_path=TON_IOT_CSV,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_cache=not args.no_cache,
    )
    logger.info(
        f"Training done — best epoch: {train_results['best_epoch']}, "
        f"val loss: {train_results['val_loss']:.6f}, "
        f"sep ratio: {train_results['separation_ratio']:.2f}x"
    )

    # ── Step 2: Load dataset and scaler produced by training ──────────────
    logger.info("\n" + "=" * 55)
    logger.info("STEP 2 — ENROLLMENT")
    logger.info("=" * 55)
    window_ds = WindowDataset.load(WINDOW_CACHE)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    dna_map = enroll_all_devices(
        window_ds, scaler, checkpoint_path=BEST_MODEL_PT
    )
    logger.info(f"Enrolled {len(dna_map)} devices")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("STEP 3 — EVALUATION")
    logger.info("=" * 55)
    report = evaluate_model(
        window_ds, dna_map, scaler,
        checkpoint_path=BEST_MODEL_PT,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info("PIPELINE COMPLETE")
    logger.info(sep)
    logger.info(f"  Total time       : {elapsed / 60:.1f} min")
    logger.info(f"  Best epoch       : {train_results['best_epoch']}")
    logger.info(f"  Intra dist       : {train_results['intra_mean']:.4f}")
    logger.info(f"  Inter dist       : {train_results['inter_mean']:.4f}")
    logger.info(f"  Sep ratio        : {train_results['separation_ratio']:.2f}x")
    logger.info(f"  ROC-AUC          : {report.roc_auc:.4f}")
    logger.info(f"  TPR  (k=2.5)     : {report.confusion_2_5.tpr * 100:.2f}%")
    logger.info(f"  FPR  (k=2.5)     : {report.confusion_2_5.fpr * 100:.2f}%")
    logger.info(f"  F1   (k=2.5)     : {report.confusion_2_5.f1:.4f}")
    logger.info(sep)

    if not args.no_plot:
        plot_roc_curve(report)


if __name__ == "__main__":
    main()
