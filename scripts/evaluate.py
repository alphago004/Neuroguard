"""
Full NEUROGUARD evaluation pipeline.

Loads the trained model and scaler, enrolls all devices using enroll_normal
windows, scores test_normal and attack windows, and prints the evaluation
report. Saves a ROC curve PNG to models/checkpoints/roc_curve.png.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pt
    python scripts/evaluate.py --no-cache   # rebuild window dataset from CSV
"""

import sys
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.training.dataset import WindowDataset, build_windows
from src.detection.enroll import enroll_all_devices
from src.training.metrics import evaluate_model, plot_roc_curve

TON_IOT_CSV    = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
WINDOW_CACHE   = PROJECT_ROOT / "data" / "processed" / "window_dataset.pkl"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
SCALER_PKL     = CHECKPOINT_DIR / "scaler.pkl"
BEST_MODEL_PT  = CHECKPOINT_DIR / "best_model.pt"


def load_dataset(use_cache: bool) -> WindowDataset:
    if use_cache and WINDOW_CACHE.exists():
        logger.info(f"Loading window dataset from cache: {WINDOW_CACHE}")
        return WindowDataset.load(WINDOW_CACHE)
    logger.info("Building window dataset from CSV…")
    ds = build_windows(TON_IOT_CSV)
    WINDOW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds.save(WINDOW_CACHE)
    return ds


def load_scaler(path: Path):
    if not path.exists():
        logger.error(f"Scaler not found at {path}. Run training first.")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def print_report(report) -> None:
    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info("NEUROGUARD — EVALUATION REPORT")
    logger.info(sep)
    logger.info(f"  Enrolled devices : {len(report.enrolled_devices)}")
    logger.info(f"  Normal windows   : {report.n_normal}")
    logger.info(f"  Attack windows   : {report.n_attack}")
    logger.info(sep)
    logger.info(f"  ROC-AUC          : {report.roc_auc:.4f}")
    logger.info(f"\n  k = 2.5 threshold")
    logger.info(f"    TPR (Detection): {report.confusion_2_5.tpr * 100:.2f}%")
    logger.info(f"    FPR (False pos): {report.confusion_2_5.fpr * 100:.2f}%")
    logger.info(f"    Precision      : {report.confusion_2_5.precision:.4f}")
    logger.info(f"    F1             : {report.confusion_2_5.f1:.4f}")
    logger.info(f"\n  k = 3.0 threshold")
    logger.info(f"    TPR (Detection): {report.confusion_3_0.tpr * 100:.2f}%")
    logger.info(f"    FPR (False pos): {report.confusion_3_0.fpr * 100:.2f}%")
    logger.info(f"    F1             : {report.confusion_3_0.f1:.4f}")

    if report.per_type:
        logger.info(f"\n  Per-attack-type detection (k=2.5):")
        logger.info(f"    {'Type':<18} {'Windows':>7}  {'Detected':>8}  {'Rate':>6}")
        logger.info(f"    {'-'*44}")
        for atype, m in sorted(report.per_type.items()):
            logger.info(
                f"    {atype:<18} {m.n_windows:>7}  {m.n_detected:>8}  "
                f"{m.detection_rate * 100:>5.1f}%"
            )

    logger.info(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NEUROGUARD")
    parser.add_argument(
        "--checkpoint", type=Path, default=BEST_MODEL_PT,
        help="Path to trained model checkpoint (default: best_model.pt)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Rebuild window dataset from CSV instead of loading cache",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip ROC curve plot",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Run training first: python -m src.training.train")
        sys.exit(1)

    window_ds = load_dataset(use_cache=not args.no_cache)
    scaler    = load_scaler(SCALER_PKL)

    logger.info("Enrolling devices…")
    dna_map = enroll_all_devices(
        window_ds, scaler, checkpoint_path=args.checkpoint
    )

    logger.info("Running evaluation…")
    report = evaluate_model(
        window_ds, dna_map, scaler,
        checkpoint_path=args.checkpoint,
    )

    print_report(report)

    if not args.no_plot:
        plot_roc_curve(report)


if __name__ == "__main__":
    main()
