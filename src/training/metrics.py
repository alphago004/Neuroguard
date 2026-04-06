"""
NEUROGUARD — Evaluation metrics for the zero-day detection paper.

This module implements the full evaluation protocol from CLAUDE.md §11:

  Step 6 — Score df_attack through the trained, enrolled system
  Step 7 — Compute: Detection Rate (TPR), FPR, ROC-AUC, detection latency

Public API
----------
  evaluate_model(window_ds, dna_map, scaler, csv_path) → EvaluationReport
  plot_roc_curve(report, output_path)

EvaluationReport fields
-----------------------
  roc_auc          : float
  scores_normal    : list[float]   — anomaly scores for all test-normal windows
  scores_attack    : list[float]   — anomaly scores for all attack windows
  labels           : list[int]     — 0=normal, 1=attack  (ground truth)
  predictions_2_5  : list[int]     — predicted labels at k_sigma=2.5
  predictions_3_0  : list[int]     — predicted labels at k_sigma=3.0
  per_type_results : dict[str, TypeMetrics]
  attack_type_map  : dict[(device_id, window_idx), str]

Attack-type reconstruction
---------------------------
WindowRecord has `flow_start` (row index within that device's flows).
We re-read the CSV once, group by src_ip, and for each attack window
determine the majority attack type across its 50 flows. This is O(N)
and done once at evaluation time.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.training.dataset import WindowDataset, WindowRecord, LABEL_NORMAL, LABEL_ATTACK
from src.detection.enroll import DeviceDNA, enroll_all_devices, CHECKPOINT_DIR
from src.detection.scorer import score_records, AnomalyResult, STATUS_ALERT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TON_IOT_CSV  = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
WINDOW_SIZE  = 50

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TypeMetrics:
    """Detection metrics for a single attack type."""
    attack_type:    str
    n_windows:      int
    n_detected:     int
    detection_rate: float      # TPR for this attack type
    mean_score:     float
    std_score:      float
    min_score:      float
    max_score:      float


@dataclass
class ConfusionStats:
    """Confusion matrix and derived metrics at one threshold."""
    k_sigma:    float
    tp:         int
    fp:         int
    tn:         int
    fn:         int
    tpr:        float   # sensitivity / recall / detection rate
    fpr:        float   # false alarm rate
    precision:  float
    f1:         float
    accuracy:   float


@dataclass
class EvaluationReport:
    """Full evaluation results — all numbers needed for the paper."""
    roc_auc:          float
    scores_normal:    list[float]
    scores_attack:    list[float]
    labels_all:       list[int]          # ground truth (0=normal, 1=attack)
    scores_all:       list[float]        # raw anomaly scores (unnormalized by threshold)
    fpr_curve:        np.ndarray         # for ROC plot
    tpr_curve:        np.ndarray         # for ROC plot
    thresholds_curve: np.ndarray         # decision thresholds along ROC
    confusion_2_5:    ConfusionStats
    confusion_3_0:    ConfusionStats
    per_type:         dict[str, TypeMetrics]
    n_normal:         int
    n_attack:         int
    enrolled_devices: list[str]


# ---------------------------------------------------------------------------
# Attack-type reconstruction from CSV
# ---------------------------------------------------------------------------

def build_attack_type_map(
    csv_path: Path,
    attack_records: list[WindowRecord],
) -> dict[tuple[str, int], str]:
    """Map each attack WindowRecord to its majority attack type.

    Reads the CSV once, groups rows by src_ip, and for each attack window
    (identified by device_id + flow_start) takes the mode of the `type`
    column across the 50 flows in that window.

    Args:
        csv_path:       Path to train_test_network.csv.
        attack_records: Attack-labeled WindowRecords from WindowDataset.

    Returns:
        Dict mapping (device_id, window_idx) → attack_type_string.
        e.g. ('192.168.1.193', 4) → 'backdoor'
    """
    logger.info("Building attack-type map from CSV…")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Group rows by src_ip, preserving order
    ip_groups: dict[str, pd.DataFrame] = {
        ip: grp.reset_index(drop=True)
        for ip, grp in df.groupby("src_ip", sort=False)
    }

    type_map: dict[tuple[str, int], str] = {}
    for record in attack_records:
        device_df = ip_groups.get(record.device_id)
        if device_df is None:
            type_map[(record.device_id, record.window_idx)] = "unknown"
            continue

        start = record.flow_start
        end   = min(start + WINDOW_SIZE, len(device_df))
        window_types = device_df.iloc[start:end]["type"].tolist()

        # Remove 'normal' rows (mixed windows) and take mode of attack types
        attack_types = [t for t in window_types if t != "normal"]
        if not attack_types:
            # Pure-normal window tagged as attack — shouldn't happen
            type_map[(record.device_id, record.window_idx)] = "normal_mislabel"
            continue

        # Mode (most frequent attack type in window)
        counts: dict[str, int] = {}
        for t in attack_types:
            counts[t] = counts.get(t, 0) + 1
        majority_type = max(counts, key=lambda k: counts[k])
        # Normalize: 'ran' is a data entry error for 'ransomware'
        if majority_type == "ran":
            majority_type = "ransomware"
        type_map[(record.device_id, record.window_idx)] = majority_type

    unique_types = set(type_map.values())
    logger.info(
        f"Attack-type map built: {len(type_map)} windows, "
        f"types: {sorted(unique_types)}"
    )
    return type_map


# ---------------------------------------------------------------------------
# Scale records helper
# ---------------------------------------------------------------------------

def _scale_records(records: list[WindowRecord], scaler) -> list[WindowRecord]:
    if not records:
        return []
    X = np.stack([r.features for r in records])
    X_s = scaler.transform(X).astype(np.float32)
    return [
        WindowRecord(r.device_id, X_s[i], r.label, r.window_idx, r.flow_start)
        for i, r in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# Confusion matrix at a given k_sigma threshold
# ---------------------------------------------------------------------------

def _confusion_at_k(
    results_normal: list[AnomalyResult],
    results_attack:  list[AnomalyResult],
    dna_map:         dict[str, DeviceDNA],
    k_sigma:         float,
) -> ConfusionStats:
    """Recompute confusion matrix using k_sigma-recalculated thresholds.

    Instead of using the stored threshold, we recompute threshold =
    mean_dist + k_sigma * std_dist from the DNA's embedding_distances,
    then re-evaluate each score against the new threshold.
    """
    def _threshold_for(dna: DeviceDNA, k: float) -> float:
        dists = dna.embedding_distances
        return float(
            np.clip(
                dists.mean() + k * dists.std(),
                0.05, 0.95
            )
        )

    tp = fp = tn = fn = 0

    for r in results_normal:
        dna = dna_map.get(r.device_id)
        if dna is None:
            continue
        thr = _threshold_for(dna, k_sigma)
        # raw_distance is stored in AnomalyResult
        predicted_alert = r.raw_distance >= thr
        if predicted_alert:
            fp += 1
        else:
            tn += 1

    for r in results_attack:
        dna = dna_map.get(r.device_id)
        if dna is None:
            continue
        thr = _threshold_for(dna, k_sigma)
        predicted_alert = r.raw_distance >= thr
        if predicted_alert:
            tp += 1
        else:
            fn += 1

    total_pos = tp + fn
    total_neg = tn + fp

    tpr       = tp / total_pos if total_pos > 0 else 0.0
    fpr       = fp / total_neg if total_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return ConfusionStats(
        k_sigma=k_sigma, tp=tp, fp=fp, tn=tn, fn=fn,
        tpr=tpr, fpr=fpr, precision=precision, f1=f1, accuracy=accuracy,
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    window_ds:       WindowDataset,
    dna_map:         dict[str, DeviceDNA],
    scaler,
    csv_path:        Path = TON_IOT_CSV,
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pt",
) -> EvaluationReport:
    """Run the full evaluation protocol and return an EvaluationReport.

    Args:
        window_ds:       WindowDataset with test_normal and attack_records.
        dna_map:         Dict of enrolled DeviceDNA objects.
        scaler:          Fitted RobustScaler.
        csv_path:        Path to raw CSV (for attack-type reconstruction).
        checkpoint_path: Trained model checkpoint.

    Returns:
        EvaluationReport with all paper metrics.
    """
    enrolled_ids = set(dna_map.keys())

    # ── Scale records ──────────────────────────────────────────────────────
    logger.info("Scaling test_normal and attack records…")
    scaled_normal = _scale_records(window_ds.test_normal, scaler)
    scaled_attack  = _scale_records(window_ds.attack_records, scaler)

    # Filter to only devices with enrolled DNA
    normal_eligible = [r for r in scaled_normal if r.device_id in enrolled_ids]
    attack_eligible  = [r for r in scaled_attack if r.device_id in enrolled_ids]

    logger.info(
        f"Evaluation set: {len(normal_eligible)} normal windows, "
        f"{len(attack_eligible)} attack windows, "
        f"{len(enrolled_ids)} enrolled devices"
    )

    # ── Score all windows ──────────────────────────────────────────────────
    logger.info("Scoring normal windows…")
    results_normal = score_records(
        normal_eligible, dna_map,
        checkpoint_path=checkpoint_path,
        compute_attribution=False,
    )

    logger.info("Scoring attack windows…")
    results_attack = score_records(
        attack_eligible, dna_map,
        checkpoint_path=checkpoint_path,
        compute_attribution=False,
    )

    # ── Build score arrays ─────────────────────────────────────────────────
    scores_normal = [r.raw_distance for r in results_normal]
    scores_attack  = [r.raw_distance for r in results_attack]

    labels_all = [0] * len(scores_normal) + [1] * len(scores_attack)
    scores_all = scores_normal + scores_attack

    # ── ROC-AUC ───────────────────────────────────────────────────────────
    logger.info("Computing ROC-AUC…")
    roc_auc = roc_auc_score(labels_all, scores_all)
    fpr_curve, tpr_curve, thresholds_curve = roc_curve(labels_all, scores_all)

    # ── Confusion matrices at k=2.5 and k=3.0 ─────────────────────────────
    logger.info("Computing confusion matrices…")
    conf_2_5 = _confusion_at_k(results_normal, results_attack, dna_map, k_sigma=2.5)
    conf_3_0 = _confusion_at_k(results_normal, results_attack, dna_map, k_sigma=3.0)

    # ── Per-attack-type breakdown ──────────────────────────────────────────
    logger.info("Building attack-type map (reads CSV once)…")
    type_map = build_attack_type_map(csv_path, window_ds.attack_records)

    # Map each scored attack result back to its attack type
    # results_attack[i] corresponds to attack_eligible[i]
    type_to_results: dict[str, list[AnomalyResult]] = {}
    for record, result in zip(attack_eligible, results_attack):
        atype = type_map.get((record.device_id, record.window_idx), "unknown")
        type_to_results.setdefault(atype, []).append(result)

    # Recompute per-type detection using k=2.5 threshold
    per_type: dict[str, TypeMetrics] = {}
    for atype, type_results in sorted(type_to_results.items()):
        scores = []
        detected = 0
        for r in type_results:
            dna = dna_map.get(r.device_id)
            if dna is None:
                continue
            dists = dna.embedding_distances
            thr = float(np.clip(dists.mean() + 2.5 * dists.std(), 0.05, 0.95))
            scores.append(r.raw_distance)
            if r.raw_distance >= thr:
                detected += 1

        if not scores:
            continue
        per_type[atype] = TypeMetrics(
            attack_type=atype,
            n_windows=len(scores),
            n_detected=detected,
            detection_rate=detected / len(scores),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
        )

    return EvaluationReport(
        roc_auc=roc_auc,
        scores_normal=scores_normal,
        scores_attack=scores_attack,
        labels_all=labels_all,
        scores_all=scores_all,
        fpr_curve=fpr_curve,
        tpr_curve=tpr_curve,
        thresholds_curve=thresholds_curve,
        confusion_2_5=conf_2_5,
        confusion_3_0=conf_3_0,
        per_type=per_type,
        n_normal=len(results_normal),
        n_attack=len(results_attack),
        enrolled_devices=sorted(enrolled_ids),
    )


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curve(
    report:      EvaluationReport,
    output_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "roc_curve.png",
) -> None:
    """Save a publication-quality ROC curve plot.

    Args:
        report:      EvaluationReport from evaluate_model().
        output_path: Destination PNG path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless backend — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed — skipping ROC plot")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Main ROC curve
    ax.plot(
        report.fpr_curve, report.tpr_curve,
        color="#2563EB", linewidth=2.5,
        label=f"NEUROGUARD (AUC = {report.roc_auc:.4f})",
        zorder=3,
    )

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=1.2,
            linestyle="--", label="Random classifier (AUC = 0.50)", zorder=2)

    # Mark operating points at k=2.5 and k=3.0
    c25 = report.confusion_2_5
    c30 = report.confusion_3_0
    ax.scatter(
        [c25.fpr], [c25.tpr],
        color="#DC2626", s=80, zorder=5,
        label=f"k=2.5  TPR={c25.tpr:.3f}  FPR={c25.fpr:.3f}",
    )
    ax.scatter(
        [c30.fpr], [c30.tpr],
        color="#16A34A", s=80, zorder=5,
        label=f"k=3.0  TPR={c30.tpr:.3f}  FPR={c30.fpr:.3f}",
    )

    # Annotations
    ax.annotate(
        f"k=2.5", xy=(c25.fpr, c25.tpr),
        xytext=(c25.fpr + 0.06, c25.tpr - 0.06),
        fontsize=9, color="#DC2626",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0),
    )
    ax.annotate(
        f"k=3.0", xy=(c30.fpr, c30.tpr),
        xytext=(c30.fpr + 0.06, c30.tpr + 0.04),
        fontsize=9, color="#16A34A",
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.0),
    )

    # Shaded area under curve
    ax.fill_between(
        report.fpr_curve, report.tpr_curve,
        alpha=0.08, color="#2563EB",
    )

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(
        "NEUROGUARD — Zero-Day IoT Anomaly Detection\nROC Curve (TON_IoT Dataset)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Add dataset stats as text box
    stats_text = (
        f"Test set: {report.n_normal} normal + {report.n_attack} attack windows\n"
        f"Devices enrolled: {len(report.enrolled_devices)}\n"
        f"Attack types: {len(report.per_type)}"
    )
    ax.text(
        0.02, 0.76, stats_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3F4F6", alpha=0.8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved → {output_path}")
