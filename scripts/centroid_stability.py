"""
Step 4 (M4) — Centroid Stability vs. Enrollment Size.

Research question: Does the Transformer component of the encoder produce
more stable device centroids, especially when enrollment data is scarce?

Method:
  - For device 192.168.1.152 (698 train_normal windows — largest pool),
    subsample N windows B=100 times (bootstrap with replacement),
    compute the embedding centroid each time,
    then measure mean pairwise cosine distance between bootstrap centroids.
  - Lower mean pairwise distance = more stable/consistent centroid.
  - Repeat for N ∈ {5, 10, 15, 20, 30, 50, 100}.
  - Compare full model (best_model.pt) vs. A1 ablation (no Transformer,
    ablation_a1_no_transformer.pt).

Expected result:
  - At small N (5–15), the Transformer encoder produces significantly
    lower inter-bootstrap centroid variance (more stable fingerprint).
  - At large N (≥50), both models converge (sufficient data averages out).
  - This justifies Contribution 2 in the paper.

Outputs:
  paper/centroid_stability.png       (300 DPI, IEEE single-column)
  centroid_stability_results.txt     (numerical table)

Usage:
    python3 scripts/centroid_stability.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from loguru import logger

from src.training.dataset import WindowDataset
from src.models.encoder import BehavioralEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH  = ROOT / "data" / "processed" / "window_dataset.pkl"
SCALER_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"
CHECKPOINT_FULL = ROOT / "models" / "checkpoints" / "best_model.pt"
CHECKPOINT_A1   = ROOT / "models" / "checkpoints" / "ablation_a1_no_transformer.pt"
FIGURE_OUT  = ROOT / "paper" / "centroid_stability.png"
RESULTS_OUT = ROOT / "centroid_stability_results.txt"

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
TARGET_DEVICE = "192.168.1.152"   # 698 train_normal + 122 enroll_normal windows
N_VALUES      = [5, 10, 15, 20, 30, 50, 100]
N_BOOTSTRAP   = 100               # bootstrap replicates per N
EMBEDDING_DIM = 64
SEED_BASE     = 42

COLORS = {
    "full":  "#0072B2",   # deep blue
    "a1":    "#D55E00",   # red-orange
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _EncoderNoTransformer(BehavioralEncoder):
    """BehavioralEncoder with TemporalEncoder bypassed — mirrors A1 ablation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)       # (batch, 128)
        x = self.layer2(x)       # (batch, 256)
        # temporal_encoder intentionally skipped
        x = self.projection(x)   # (batch, 64)
        return x


def load_encoder(checkpoint_path: Path, device: torch.device,
                 use_transformer: bool = True) -> BehavioralEncoder:
    """Load encoder weights from a SiameseNetwork checkpoint."""
    if use_transformer:
        encoder = BehavioralEncoder(input_dim=60, embedding_dim=EMBEDDING_DIM,
                                    transformer_layers=1).to(device)
    else:
        # A1 ablation: same weights but forward() skips temporal_encoder
        encoder = _EncoderNoTransformer(input_dim=60, embedding_dim=EMBEDDING_DIM,
                                        transformer_layers=1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # SiameseNetwork checkpoints store encoder under 'encoder.*' keys
    enc_state = {k[len("encoder."):]: v for k, v in state.items()
                 if k.startswith("encoder.")}
    if not enc_state:
        enc_state = state
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        logger.warning(f"Missing keys in {checkpoint_path.name}: {missing[:3]}…")
    return encoder


def get_embeddings(encoder: BehavioralEncoder, features: np.ndarray,
                   device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run encoder in eval mode, return (N, 64) embedding matrix."""
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i:i+batch_size]).to(device)
            emb = encoder(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalized centroid vectors."""
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(an, bn))


def bootstrap_centroid_variance(embeddings: np.ndarray, n: int,
                                n_bootstrap: int, rng: np.random.Generator) -> float:
    """
    For each of B bootstrap trials, sample n embeddings → compute centroid.
    Return mean pairwise cosine distance between bootstrap centroids.
    """
    centroids = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(embeddings), size=n, replace=True)
        centroid = embeddings[idx].mean(axis=0)
        centroids.append(centroid)

    # Mean pairwise cosine distance between centroids
    centroids = np.stack(centroids)
    dists = []
    for i in range(n_bootstrap):
        for j in range(i + 1, min(i + 10, n_bootstrap)):  # sample pairs for speed
            dists.append(cosine_distance(centroids[i], centroids[j]))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_stability_experiment(
    features: np.ndarray,
    encoder: BehavioralEncoder,
    device: torch.device,
    label: str,
) -> dict[int, tuple[float, float]]:
    """
    For each N in N_VALUES, bootstrap B times to estimate centroid stability.
    Returns {N: (mean_dist, std_dist)} across bootstrap trials.
    """
    logger.info(f"Computing embeddings for {label} encoder…")
    embeddings = get_embeddings(encoder, features, device)
    logger.info(f"  {label}: embeddings shape = {embeddings.shape}")

    results = {}
    rng = np.random.default_rng(SEED_BASE)

    for n in N_VALUES:
        if n > len(embeddings):
            logger.warning(f"  {label}: N={n} > available ({len(embeddings)}), skipping")
            continue
        # Run 5 independent bootstrap runs and average for std estimate
        run_means = []
        for run in range(5):
            rng_run = np.random.default_rng(SEED_BASE + run * 100)
            mean_dist = bootstrap_centroid_variance(embeddings, n, N_BOOTSTRAP, rng_run)
            run_means.append(mean_dist)
        results[n] = (float(np.mean(run_means)), float(np.std(run_means)))
        logger.info(f"  {label}: N={n:3d}  mean_centroid_dist={results[n][0]:.4f} ± {results[n][1]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_stability(
    results_full: dict[int, tuple[float, float]],
    results_a1:   dict[int, tuple[float, float]],
    n_pool: int,
    out_path: Path,
) -> None:
    """Plot centroid stability vs. enrollment size — IEEE single-column."""
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "lines.linewidth":  1.8,
        "figure.dpi":       300,
    })

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    def plot_method(results, color, label, marker):
        ns    = sorted(results.keys())
        means = [results[n][0] for n in ns]
        stds  = [results[n][1] for n in ns]
        ax.plot(ns, means, color=color, marker=marker, markersize=5,
                label=label, zorder=3)
        ax.fill_between(ns,
                        [max(m - s, 0) for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15, zorder=2)

    plot_method(results_full, COLORS["full"],
                "NEUROGUARD (Siamese + Transformer)", "o")
    plot_method(results_a1,   COLORS["a1"],
                "Ablation A1 (no Transformer)",       "s")

    ax.set_xlabel("Enrollment pool size $N$ (windows)")
    ax.set_ylabel("Mean centroid instability\n(inter-bootstrap cosine distance)")
    ax.set_title("Centroid Stability vs. Enrollment Size")
    ax.legend(loc="center right", framealpha=0.92, edgecolor="#CCCCCC",
              handlelength=2.0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=0, right=max(N_VALUES) + 5)
    ax.set_ylim(bottom=0)

    # Note pool size in subtitle text (inside plot bounds)
    ax.text(0.98, 0.96, f"Device pool: {n_pool} normal windows",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="#555555", style="italic")

    fig.tight_layout(pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Centroid stability figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Centroid Stability Experiment (M4) ===")

    device = get_device()
    logger.info(f"Torch device: {device}")

    # Load dataset
    window_ds = WindowDataset.load(CACHE_PATH)
    logger.info(f"Dataset loaded: {len(window_ds.records):,} total windows")

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Get all normal windows for target device (train + enroll + test normal)
    all_normal = (window_ds.train_normal
                  + window_ds.enroll_normal
                  + window_ds.test_normal)
    target_normal = [r for r in all_normal if r.device_id == TARGET_DEVICE]

    if not target_normal:
        logger.error(f"No normal windows found for device {TARGET_DEVICE}")
        sys.exit(1)

    n_pool = len(target_normal)
    logger.info(f"Device {TARGET_DEVICE}: {n_pool} normal windows available")

    # Scale features
    X_raw = np.stack([r.features for r in target_normal]).astype(np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Cap N_VALUES to available pool size
    valid_Ns = [n for n in N_VALUES if n <= n_pool]
    logger.info(f"Valid N values (≤ {n_pool}): {valid_Ns}")

    # ── Load encoders ─────────────────────────────────────────────────────────
    logger.info(f"Loading full encoder from {CHECKPOINT_FULL.name}…")
    enc_full = load_encoder(CHECKPOINT_FULL, device, use_transformer=True)

    # A1 ablation: same architecture but forward() bypasses temporal_encoder
    logger.info(f"Loading A1 encoder from {CHECKPOINT_A1.name}…")
    enc_a1 = load_encoder(CHECKPOINT_A1, device, use_transformer=False)

    # ── Run stability experiment ──────────────────────────────────────────────
    results_full = run_stability_experiment(X_scaled, enc_full, device, "FULL")
    results_a1   = run_stability_experiment(X_scaled, enc_a1,   device, "A1")

    # ── Print comparison table ────────────────────────────────────────────────
    lines = []
    lines.append("CENTROID STABILITY EXPERIMENT RESULTS")
    lines.append(f"Device: {TARGET_DEVICE}  |  Pool size: {n_pool} normal windows")
    lines.append(f"Bootstrap replicates: {N_BOOTSTRAP} per N")
    lines.append("")
    lines.append(f"{'N':>5}  {'FULL (mean±std)':>22}  {'A1 (mean±std)':>22}  {'Δ (A1−FULL)':>12}  {'Improvement':>12}")
    lines.append("-" * 82)

    for n in valid_Ns:
        if n not in results_full or n not in results_a1:
            continue
        fm, fs = results_full[n]
        am, as_ = results_a1[n]
        delta = am - fm
        pct = (delta / am * 100) if am > 0 else 0.0
        lines.append(
            f"{n:>5}  {fm:.4f} ± {fs:.4f}          "
            f"{am:.4f} ± {as_:.4f}          "
            f"{delta:>+.4f}          {pct:>+.1f}%"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Lower value = more stable centroid = less bootstrap variance.")
    lines.append("  Δ > 0 means FULL model is more stable than A1 at this N.")

    report = "\n".join(lines)
    print(report)

    with open(RESULTS_OUT, "w") as f:
        f.write(report + "\n")
    logger.info(f"Results saved → {RESULTS_OUT}")

    # ── Generate figure ───────────────────────────────────────────────────────
    plot_stability(results_full, results_a1, n_pool, FIGURE_OUT)

    logger.info("Done.")


if __name__ == "__main__":
    main()
