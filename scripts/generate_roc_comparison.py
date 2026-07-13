"""
Step 3 (M1) — Baseline ROC Comparison Figure.

Generates a publication-quality ROC curve overlay comparing NEUROGUARD
against three per-device baselines:
  - IsolationForest (per-device)
  - OneClassSVM (per-device)
  - Autoencoder (per-device)

The figure includes:
  - 4 ROC curves with AUC in legend
  - Random-chance diagonal
  - Operating-point markers at each method's threshold
  - Inset zoomed to FPR ∈ [0, 0.15] to show low-FPR discriminability
  - Shaded target zone (FPR < 0.02)

Outputs:
  paper/roc_comparison.png   (300 DPI, IEEE-compatible single-column)
  models/checkpoints/raw_roc_scores.pkl  (raw scores for reproducibility)

Usage:
    python3 scripts/generate_roc_comparison.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler
from loguru import logger

from src.training.dataset import WindowDataset
from src.detection.enroll import enroll_all_devices
from src.training.metrics import evaluate_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH      = ROOT / "data" / "processed" / "window_dataset.pkl"
SCALER_PATH     = ROOT / "models" / "checkpoints" / "scaler.pkl"
CHECKPOINT      = ROOT / "models" / "checkpoints" / "best_model.pt"
RAW_SCORES_OUT  = ROOT / "models" / "checkpoints" / "raw_roc_scores.pkl"
FIGURE_OUT      = ROOT / "paper" / "roc_comparison.png"

MIN_TRAIN_WINDOWS = 5

# Wong's colorblind-friendly palette (widely used in ML papers)
COLORS = {
    "neuroguard":     "#0072B2",   # deep blue
    "iforest":        "#009E73",   # green
    "ocsvm":          "#E69F00",   # amber
    "autoencoder":    "#D55E00",   # red-orange
    "chance":         "#CCCCCC",   # light grey
}


# ---------------------------------------------------------------------------
# Autoencoder (same architecture as run_baselines_per_device.py)
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(60, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                                     nn.Linear(32, 60))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _ae_recon_errors(ae: Autoencoder, X: np.ndarray) -> np.ndarray:
    ae.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32))
        recon = ae(t).numpy()
    return np.mean((X - recon) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Baseline evaluation (per-device)
# ---------------------------------------------------------------------------

def run_baselines(window_ds: WindowDataset, scaler: RobustScaler) -> dict:
    """Re-run all three baselines and return raw per-window scores."""
    logger.info("Running per-device baselines to collect raw scores…")

    scores_if_normal:  list[float] = []
    scores_if_attack:  list[float] = []
    scores_svm_normal: list[float] = []
    scores_svm_attack: list[float] = []
    scores_ae_normal:  list[float] = []
    scores_ae_attack:  list[float] = []

    # Operating-point thresholds (p95 of each device's training normal scores)
    thresholds_if:  list[float] = []
    thresholds_svm: list[float] = []
    thresholds_ae:  list[float] = []

    # Get enrolled device IDs (same set as NEUROGUARD)
    enrolled_ids = {r.device_id for r in window_ds.enroll_normal}

    for device_id in sorted(window_ds.device_ids):
        if device_id not in enrolled_ids:
            continue

        train = [r for r in window_ds.train_normal if r.device_id == device_id]
        test_n = [r for r in window_ds.test_normal  if r.device_id == device_id]
        attack = [r for r in window_ds.attack_records if r.device_id == device_id]

        # Require training data and test_normal for FPR evaluation.
        # Do NOT require attack records here — normal-only devices must still
        # contribute to FPR evaluation so the baseline negative set matches
        # NEUROGUARD's 109-window test_normal pool (N3 fix).
        if len(train) < MIN_TRAIN_WINDOWS or not test_n:
            continue

        X_train  = scaler.transform(np.stack([r.features for r in train])).astype(np.float32)
        X_test_n = scaler.transform(np.stack([r.features for r in test_n])).astype(np.float32)

        # ── IsolationForest ────────────────────────────────────────────────
        iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iforest.fit(X_train)
        s_n = -iforest.score_samples(X_test_n)
        scores_if_normal.extend(s_n.tolist())
        thresholds_if.append(float(np.percentile(s_n, 95)))
        if attack:
            X_attack = scaler.transform(np.stack([r.features for r in attack])).astype(np.float32)
            scores_if_attack.extend((-iforest.score_samples(X_attack)).tolist())

        # ── OneClassSVM ────────────────────────────────────────────────────
        ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        ocsvm.fit(X_train)
        s_n = -ocsvm.decision_function(X_test_n)
        scores_svm_normal.extend(s_n.tolist())
        thresholds_svm.append(float(np.percentile(s_n, 95)))
        if attack:
            X_attack = scaler.transform(np.stack([r.features for r in attack])).astype(np.float32)
            scores_svm_attack.extend((-ocsvm.decision_function(X_attack)).tolist())

        # ── Autoencoder ────────────────────────────────────────────────────
        ae = Autoencoder()
        opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(60):
            ae.train()
            idx = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), 32):
                batch = torch.from_numpy(X_train[idx[i:i+32]])
                opt.zero_grad()
                loss = nn.functional.mse_loss(ae(batch), batch)
                loss.backward()
                opt.step()
        s_n = _ae_recon_errors(ae, X_test_n)
        scores_ae_normal.extend(s_n.tolist())
        thresholds_ae.append(float(np.percentile(s_n, 95)))
        if attack:
            X_attack = scaler.transform(np.stack([r.features for r in attack])).astype(np.float32)
            scores_ae_attack.extend(_ae_recon_errors(ae, X_attack).tolist())

        logger.debug(f"  {device_id}: IF/SVM/AE baselines computed (attack={len(attack)})")

    logger.info(
        f"Baselines done: {len(scores_if_normal)} normal, "
        f"{len(scores_if_attack)} attack windows"
    )

    return {
        "IsolationForest": {
            "scores_normal": np.array(scores_if_normal),
            "scores_attack": np.array(scores_if_attack),
            "op_threshold":  float(np.mean(thresholds_if)),
        },
        "OneClassSVM": {
            "scores_normal": np.array(scores_svm_normal),
            "scores_attack": np.array(scores_svm_attack),
            "op_threshold":  float(np.mean(thresholds_svm)),
        },
        "Autoencoder": {
            "scores_normal": np.array(scores_ae_normal),
            "scores_attack": np.array(scores_ae_attack),
            "op_threshold":  float(np.mean(thresholds_ae)),
        },
    }


# ---------------------------------------------------------------------------
# NEUROGUARD evaluation
# ---------------------------------------------------------------------------

def run_neuroguard(window_ds: WindowDataset, scaler: RobustScaler) -> dict:
    """Load existing DNA and model, return raw scores + ROC curve."""
    logger.info("Running NEUROGUARD evaluation…")
    dna_map = enroll_all_devices(window_ds, scaler, checkpoint_path=CHECKPOINT)
    report  = evaluate_model(window_ds, dna_map, scaler, checkpoint_path=CHECKPOINT)

    return {
        "scores_normal": np.array(report.scores_normal),
        "scores_attack": np.array(report.scores_attack),
        "fpr_curve":     report.fpr_curve,
        "tpr_curve":     report.tpr_curve,
        "op_fpr":        report.confusion_2_5.fpr,
        "op_tpr":        report.confusion_2_5.tpr,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_roc_comparison(ng: dict, baselines: dict, out_path: Path) -> None:
    """Generate the 4-method ROC comparison figure with zoomed inset."""

    # Compute ROC curves for baselines
    roc_data = {}
    for name, data in baselines.items():
        sn = data["scores_normal"]
        sa = data["scores_attack"]
        labels = [0] * len(sn) + [1] * len(sa)
        scores = np.concatenate([sn, sa])
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find operating point: threshold closest to p95 of normal scores
        thr_op = data["op_threshold"]
        # Find the threshold index closest to op_threshold
        idx = np.searchsorted(thresholds[::-1], thr_op)
        idx = len(thresholds) - 1 - idx
        idx = min(max(idx, 0), len(fpr) - 1)
        op_fpr = float(fpr[idx])
        op_tpr = float(tpr[idx])

        roc_data[name] = {
            "auc": auc, "fpr": fpr, "tpr": tpr,
            "op_fpr": op_fpr, "op_tpr": op_tpr,
        }

    # NEUROGUARD ROC
    sn = ng["scores_normal"]
    sa = ng["scores_attack"]
    labels = [0] * len(sn) + [1] * len(sa)
    scores = np.concatenate([sn, sa])
    ng_auc = roc_auc_score(labels, scores)
    ng_fpr, ng_tpr, _ = roc_curve(labels, scores)

    logger.info("Plotting ROC comparison…")

    # ── Figure setup ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  7.5,
        "lines.linewidth":  1.5,
        "figure.dpi":       300,
    })

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # ── Shaded target FPR zone (< 2%) ────────────────────────────────────────
    ax.axvspan(0, 0.02, alpha=0.06, color="#0072B2", zorder=0)

    # ── Diagonal (chance line) ────────────────────────────────────────────────
    ax.plot([0, 1], [0, 1], color=COLORS["chance"], lw=1.0,
            linestyle="--", zorder=1, label="Random chance")

    # ── Baseline curves ────────────────────────────────────────────────────────
    method_cfg = [
        ("IsolationForest", COLORS["iforest"],     "IsolationForest", "-"),
        ("OneClassSVM",     COLORS["ocsvm"],        "OneClassSVM",      "--"),
        ("Autoencoder",     COLORS["autoencoder"],  "Autoencoder",      ":"),
    ]

    for name, color, label, ls in method_cfg:
        d = roc_data[name]
        auc_str = f"{d['auc']:.3f}"
        # Flag AUC < 0.5 (score-polarity inversion)
        note = "†" if d["auc"] < 0.5 else ""
        ax.plot(d["fpr"], d["tpr"],
                color=color, lw=1.4, linestyle=ls, zorder=3, alpha=0.85,
                label=f"{label}  (AUC={auc_str}{note})")
        # Operating point marker
        ax.scatter(d["op_fpr"], d["op_tpr"],
                   color=color, marker="D", s=30, zorder=5, edgecolors="white",
                   linewidths=0.5)

    # ── NEUROGUARD curve ──────────────────────────────────────────────────────
    ax.plot(ng_fpr, ng_tpr,
            color=COLORS["neuroguard"], lw=2.2, linestyle="-", zorder=4,
            label=f"NEUROGUARD [ours]  (AUC={ng_auc:.3f})")
    ax.scatter(ng["op_fpr"], ng["op_tpr"],
               color=COLORS["neuroguard"], marker="*", s=80, zorder=6,
               edgecolors="white", linewidths=0.5)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve Comparison — TON_IoT")
    ax.legend(loc="lower right", framealpha=0.92, edgecolor="#CCCCCC",
              handlelength=2.0, labelspacing=0.3)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_aspect("equal")

    # Add footnote for inverted methods
    fig.text(0.12, 0.01,
             "† AUC < 0.5: method assigns lower anomaly scores to attacks than normal traffic.",
             fontsize=6.5, color="#555555")

    # ── Zoomed inset: FPR ∈ [0, 0.15] ────────────────────────────────────────
    x1, x2, y1, y2 = 0.0, 0.15, 0.0, 1.02
    axins = ax.inset_axes([0.32, 0.10, 0.62, 0.48])

    # Draw inset
    axins.axvspan(0, 0.02, alpha=0.08, color="#0072B2", zorder=0)
    axins.plot([0, 0.15], [0, 0.15], color=COLORS["chance"], lw=0.8,
               linestyle="--", zorder=1)

    for name, color, label, ls in method_cfg:
        d = roc_data[name]
        axins.plot(d["fpr"], d["tpr"], color=color, lw=1.2,
                   linestyle=ls, zorder=3, alpha=0.85)
        axins.scatter(d["op_fpr"], d["op_tpr"],
                      color=color, marker="D", s=20, zorder=5,
                      edgecolors="white", linewidths=0.4)

    axins.plot(ng_fpr, ng_tpr, color=COLORS["neuroguard"], lw=1.8,
               linestyle="-", zorder=4)
    axins.scatter(ng["op_fpr"], ng["op_tpr"],
                  color=COLORS["neuroguard"], marker="*", s=55, zorder=6,
                  edgecolors="white", linewidths=0.4)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xlabel("FPR", fontsize=7)
    axins.set_ylabel("TPR", fontsize=7)
    axins.tick_params(labelsize=6.5)
    axins.grid(True, alpha=0.2, linewidth=0.4)
    axins.set_title("Zoom: FPR ∈ [0, 0.15]", fontsize=7.5, pad=2)

    # Connect inset to main axes
    ax.indicate_inset_zoom(axins, edgecolor="#888888", linewidth=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC comparison figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== ROC Comparison Generator ===")

    window_ds = WindowDataset.load(CACHE_PATH)
    logger.info(f"Dataset loaded: {len(window_ds.records):,} total windows")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded")

    # 1. NEUROGUARD
    ng_data = run_neuroguard(window_ds, scaler)

    # 2. Baselines
    baseline_data = run_baselines(window_ds, scaler)

    # 3. Save raw scores for reproducibility
    raw_scores = {"neuroguard": ng_data, "baselines": baseline_data}
    with open(RAW_SCORES_OUT, "wb") as f:
        pickle.dump(raw_scores, f)
    logger.info(f"Raw scores saved → {RAW_SCORES_OUT}")

    # 4. Print summary
    logger.info("\n--- ROC AUC Summary ---")
    sn, sa = ng_data["scores_normal"], ng_data["scores_attack"]
    logger.info(f"  NEUROGUARD:       AUC={roc_auc_score([0]*len(sn)+[1]*len(sa), np.concatenate([sn,sa])):.4f}  "
                f"op=({ng_data['op_fpr']:.3f}, {ng_data['op_tpr']:.3f})")
    for name, d in baseline_data.items():
        sn, sa = d["scores_normal"], d["scores_attack"]
        labels = [0]*len(sn)+[1]*len(sa)
        scores = np.concatenate([sn, sa])
        auc = roc_auc_score(labels, scores)
        logger.info(f"  {name:<20} AUC={auc:.4f}  op=({d['op_threshold']:.3f})")

    # 5. Generate figure
    plot_roc_comparison(ng_data, baseline_data, FIGURE_OUT)
    logger.info("Done.")


if __name__ == "__main__":
    main()
