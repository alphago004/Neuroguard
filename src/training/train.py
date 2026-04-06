"""
NEUROGUARD — Training loop for the Siamese behavioral encoder.

Entry point
-----------
    python -m src.training.train [--epochs N] [--batch-size N] [--lr F]

Or import and call train() directly from notebooks / scripts.

Training protocol
-----------------
1.  Load train_test_network.csv → build WindowDataset (or load from cache)
2.  Fit RobustScaler on train_normal feature vectors ONLY → save scaler.pkl
3.  Apply scaler to all windows (train + test + attack)
4.  Build PairDataset (train split) and PairDataset (val split from test_normal)
5.  Train for up to EPOCHS with:
      - AdamW (lr=1e-3, weight_decay=1e-4)
      - CosineAnnealingLR (T_max=EPOCHS)
      - Early stopping (patience=10 on val loss)
      - Checkpoint best model by val loss → models/checkpoints/best_model.pt
6.  After training: compute intra/inter-class distances on val set and log

RobustScaler note
-----------------
RobustScaler is fitted ONLY on train_normal feature vectors. It is then
applied to ALL windows (train, val, attack) at inference time. This is
saved to models/checkpoints/scaler.pkl and loaded by scorer.py.

The scaler is critical: raw features span orders of magnitude
(byte counts 0–10^5, ratios 0–1). Without scaling, the Transformer
attention is dominated by large-magnitude byte features and the small
protocol ratio features are invisible.

Validation set
--------------
We use test_normal windows (the held-out 20%) as the validation set
during training. This does NOT violate the evaluation protocol because:
  - These windows contain ONLY normal traffic
  - The Siamese model never sees attack labels during training
  - The test_normal set is used again during enrollment in enroll.py —
    this is intentional: enrollment uses the same distribution the model
    was validated on
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.training.dataset import (
    WindowDataset,
    PairDataset,
    WindowRecord,
    build_windows,
)
from src.models.siamese import SiameseNetwork, build_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
TON_IOT_CSV    = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BEST_MODEL_PT  = CHECKPOINT_DIR / "best_model.pt"
SCALER_PKL     = CHECKPOINT_DIR / "scaler.pkl"
WINDOW_CACHE   = PROJECT_ROOT / "data" / "processed" / "window_dataset.pkl"

# ---------------------------------------------------------------------------
# Hyperparameters (mirrors CLAUDE.md §10)
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS        = 100
DEFAULT_BATCH_SIZE    = 128
DEFAULT_LR            = 1e-3
DEFAULT_WEIGHT_DECAY  = 1e-4
DEFAULT_PATIENCE      = 10
DEFAULT_DEVICE_CAP    = 50       # stratified cap: prevents .152 (698 windows)
                                 # from dominating pair distribution. All 16
                                 # devices contribute equally at ≤50 windows.
DEFAULT_N_PAIRS       = 100_000  # pairs per epoch (sampled fresh each epoch)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return MPS, CUDA, or CPU — in that priority order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Scaler: fit on train_normal features ONLY
# ---------------------------------------------------------------------------

def fit_scaler(train_records: list[WindowRecord]) -> RobustScaler:
    """Fit a RobustScaler on the raw feature vectors of train_normal windows.

    RobustScaler uses median and IQR — resistant to the outliers that are
    common in network traffic (a single large file transfer can spike byte
    counts far above the 75th percentile).

    Args:
        train_records: List of WindowRecord from WindowDataset.train_normal.

    Returns:
        Fitted RobustScaler instance.
    """
    X = np.stack([r.features for r in train_records])   # (N, 60)
    scaler = RobustScaler()
    scaler.fit(X)
    logger.info(
        f"RobustScaler fitted on {len(train_records)} train_normal windows "
        f"(shape {X.shape})"
    )
    return scaler


def apply_scaler(
    records: list[WindowRecord],
    scaler: RobustScaler,
) -> list[WindowRecord]:
    """Return new WindowRecord list with scaled feature vectors.

    Creates new frozen dataclass instances — does not mutate originals.

    Args:
        records: WindowRecord list to transform.
        scaler:  Fitted RobustScaler.

    Returns:
        New list of WindowRecord with scaled float32 features.
    """
    if not records:
        return []
    X = np.stack([r.features for r in records])
    X_scaled = scaler.transform(X).astype(np.float32)
    return [
        WindowRecord(
            device_id=r.device_id,
            features=X_scaled[i],
            label=r.label,
            window_idx=r.window_idx,
            flow_start=r.flow_start,
        )
        for i, r in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# Distance metrics (for reporting after training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embedding_distances(
    model: SiameseNetwork,
    records: list[WindowRecord],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute intra-class and inter-class mean Euclidean distances.

    These are the two most diagnostic numbers for contrastive learning:
      intra_class_dist: mean distance between windows of the SAME device
                        → target < 0.5 (tight clusters)
      inter_class_dist: mean distance between windows of DIFFERENT devices
                        → target > 1.5 (well-separated clusters)

    A ratio inter/intra > 3.0 indicates strong discriminative power.

    Args:
        model:      Trained SiameseNetwork in eval mode.
        records:    WindowRecord list (typically val set).
        device:     Inference device.
        batch_size: Embedding batch size (unrelated to pair batch size).

    Returns:
        Dict with keys: intra_mean, inter_mean, separation_ratio.
    """
    model.eval()

    # Embed all records in batches
    all_features = np.stack([r.features for r in records])
    all_device_ids = [r.device_id for r in records]

    embeddings = []
    for start in range(0, len(all_features), batch_size):
        batch = torch.from_numpy(all_features[start:start + batch_size]).to(device)
        emb = model.encoder(batch)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)   # (N, 64)

    # Group by device
    device_to_indices: dict[str, list[int]] = {}
    for i, dev in enumerate(all_device_ids):
        device_to_indices.setdefault(dev, []).append(i)

    devices = list(device_to_indices.keys())

    # Intra-class distances: sample up to 500 pairs per device
    intra_dists = []
    rng = np.random.default_rng(42)
    for dev, indices in device_to_indices.items():
        if len(indices) < 2:
            continue
        indices = np.array(indices)
        n_pairs = min(500, len(indices) * (len(indices) - 1) // 2)
        i_idx = rng.choice(indices, n_pairs)
        j_idx = rng.choice(indices, n_pairs)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        if len(i_idx) == 0:
            continue
        dists = torch.norm(embeddings[i_idx] - embeddings[j_idx], dim=1)
        intra_dists.extend(dists.tolist())

    # Inter-class distances: sample 1000 cross-device pairs
    inter_dists = []
    n_inter = min(2000, len(devices) * (len(devices) - 1) * 10)
    for _ in range(n_inter):
        dev_a, dev_b = rng.choice(len(devices), 2, replace=False)
        idx_a = rng.choice(device_to_indices[devices[dev_a]])
        idx_b = rng.choice(device_to_indices[devices[dev_b]])
        d = torch.norm(embeddings[idx_a] - embeddings[idx_b]).item()
        inter_dists.append(d)

    intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
    inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
    ratio      = inter_mean / intra_mean if intra_mean > 0 else 0.0

    return {
        "intra_mean":       intra_mean,
        "inter_mean":       inter_mean,
        "separation_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train(
    csv_path:      Path           = TON_IOT_CSV,
    epochs:        int            = DEFAULT_EPOCHS,
    batch_size:    int            = DEFAULT_BATCH_SIZE,
    lr:            float          = DEFAULT_LR,
    weight_decay:  float          = DEFAULT_WEIGHT_DECAY,
    patience:      int            = DEFAULT_PATIENCE,
    device_cap:    int            = DEFAULT_DEVICE_CAP,
    n_pairs:       Optional[int]  = DEFAULT_N_PAIRS,
    use_cache:     bool           = True,
    save_dir:      Path           = CHECKPOINT_DIR,
) -> dict:
    """Train the Siamese encoder and return final metrics.

    Args:
        csv_path:     Path to train_test_network.csv.
        epochs:       Max training epochs.
        batch_size:   DataLoader batch size.
        lr:           Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        patience:     Early stopping patience (epochs without val improvement).
        device_cap:   Max windows per device in PairDataset.
        n_pairs:      Total pairs per dataset (None = full balanced set).
        use_cache:    Load WindowDataset from cache if available.
        save_dir:     Directory for best_model.pt and scaler.pkl.

    Returns:
        Dict with keys: train_loss, val_loss, intra_mean, inter_mean,
        separation_ratio, best_epoch, total_time_s.
    """
    t_start = time.time()
    device  = get_device()
    logger.info(f"Training device: {device}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load / build WindowDataset ─────────────────────────────────────
    if use_cache and WINDOW_CACHE.exists():
        logger.info(f"Loading WindowDataset from cache: {WINDOW_CACHE}")
        window_ds = WindowDataset.load(WINDOW_CACHE)
    else:
        logger.info("Building WindowDataset from CSV (this takes ~10s)…")
        window_ds = build_windows(csv_path)
        WINDOW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        window_ds.save(WINDOW_CACHE)

    stats = window_ds.stats()
    logger.info(
        f"Dataset: {stats['num_devices']} devices | "
        f"{stats['train_normal']} train_normal | "
        f"{stats['test_normal']} test_normal | "
        f"{stats['attack_windows']} attack windows"
    )

    # ── 2. Fit RobustScaler on train_normal ONLY ──────────────────────────
    scaler = fit_scaler(window_ds.train_normal)
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved → {save_dir / 'scaler.pkl'}")

    # ── 3. Scale all splits ───────────────────────────────────────────────
    scaled_train = apply_scaler(window_ds.train_normal, scaler)
    scaled_test  = apply_scaler(window_ds.test_normal,  scaler)

    # Patch scaled records back into a temporary WindowDataset for PairDataset
    scaled_ds = WindowDataset.__new__(WindowDataset)
    scaled_ds.records        = scaled_train + scaled_test
    scaled_ds.train_normal   = scaled_train
    scaled_ds.test_normal    = scaled_test
    scaled_ds.attack_records = apply_scaler(window_ds.attack_records, scaler)
    scaled_ds.device_ids     = window_ds.device_ids
    scaled_ds.device_to_idx  = window_ds.device_to_idx
    scaled_ds._seed          = window_ds._seed

    # ── 4. Build PairDatasets ─────────────────────────────────────────────
    val_pairs = PairDataset(
        scaled_ds, split="test",
        n_pairs=n_pairs, device_cap=device_cap, seed=0,
    )

    pair_counts = window_ds.possible_pairs("train")
    logger.info(
        f"Pairs — train pool: {pair_counts['balanced_total']:,} | "
        f"per-epoch sample: {n_pairs:,} | val fixed: {len(val_pairs):,}"
    )

    # val_loader is fixed for the entire run (deterministic evaluation)
    val_loader = DataLoader(
        val_pairs, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    # train_loader is rebuilt each epoch with a new seed so the model
    # never sees the same pair twice — this is the key anti-overfitting measure
    # for contrastive learning on small datasets.

    # ── 5. Build model ────────────────────────────────────────────────────
    # transformer_layers=1: halves Transformer params (~609k vs 1.1M).
    # margin=3.0: widens the negative-pair target zone (inter-class dist was
    # saturating at 0.85 with margin=2.0 — widening gives the loss more
    # gradient signal to keep pushing different-device embeddings apart).
    model, loss_fn = build_model(transformer_layers=1, margin=3.0)
    model   = model.to(device)
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── 6. Training loop ──────────────────────────────────────────────────
    best_val_loss   = float("inf")
    best_epoch      = 0
    epochs_no_improv = 0

    train_losses: list[float] = []
    val_losses:   list[float] = []

    logger.info(
        f"Starting training: {epochs} epochs | "
        f"batch={batch_size} | lr={lr} | patience={patience}"
    )
    logger.info(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
                f"{'LR':>10}  {'Status'}")
    logger.info("-" * 65)

    for epoch in range(1, epochs + 1):
        # Rebuild train pairs each epoch with a fresh seed derived from the
        # epoch number. This ensures every epoch sees a different sample from
        # the ~270k+ pair pool — the model cannot memorize pair assignments.
        epoch_pairs = PairDataset(
            scaled_ds, split="train",
            n_pairs=n_pairs, device_cap=device_cap, seed=epoch * 7,
        )
        train_loader = DataLoader(
            epoch_pairs, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for anchor, pair, labels in train_loader:
            anchor = anchor.to(device)
            pair   = pair.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            emb_a, emb_b = model(anchor, pair)
            dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
            loss = loss_fn(dist, labels)
            loss.backward()

            # Gradient clipping: prevents exploding gradients in Transformer
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for anchor, pair, labels in val_loader:
                anchor = anchor.to(device)
                pair   = pair.to(device)
                labels = labels.to(device)
                emb_a, emb_b = model(anchor, pair)
                dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
                loss = loss_fn(dist, labels)
                val_running += loss.item()

        val_loss = val_running / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Early stopping + checkpoint ────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improv = 0
            model.save(save_dir / "best_model.pt")
            status = "✓ saved"
        else:
            epochs_no_improv += 1
            status = f"no improv {epochs_no_improv}/{patience}"

        logger.info(
            f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
            f"{current_lr:>10.2e}  {status}"
        )

        if epochs_no_improv >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} — "
                f"no validation improvement for {patience} epochs"
            )
            break

    # ── 7. Load best model and compute distance metrics ───────────────────
    logger.info(f"\nLoading best model (epoch {best_epoch}, val={best_val_loss:.6f})")
    best_model, _ = build_model(transformer_layers=1, margin=3.0)
    best_model = best_model.to(device)
    best_model.load(save_dir / "best_model.pt", device=device)

    val_records_all = scaled_ds.test_normal
    dist_metrics = compute_embedding_distances(best_model, val_records_all, device)

    total_time = time.time() - t_start

    # ── 8. Final report ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Best epoch:          {best_epoch}")
    logger.info(f"  Final train loss:    {train_losses[-1]:.6f}")
    logger.info(f"  Best val loss:       {best_val_loss:.6f}")
    logger.info(f"  Intra-class dist:    {dist_metrics['intra_mean']:.4f}  (target < 0.5)")
    logger.info(f"  Inter-class dist:    {dist_metrics['inter_mean']:.4f}  (target > 1.5)")
    logger.info(f"  Separation ratio:    {dist_metrics['separation_ratio']:.2f}x  (target > 3.0x)")
    logger.info(f"  Total time:          {total_time:.1f}s")
    logger.info("=" * 65)

    result = {
        "train_loss":       train_losses[-1],
        "val_loss":         best_val_loss,
        "intra_mean":       dist_metrics["intra_mean"],
        "inter_mean":       dist_metrics["inter_mean"],
        "separation_ratio": dist_metrics["separation_ratio"],
        "best_epoch":       best_epoch,
        "total_time_s":     total_time,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEUROGUARD Siamese encoder")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--no-cache",    action="store_true",
                        help="Rebuild WindowDataset from scratch (ignore cache)")
    args = parser.parse_args()

    results = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_cache=not args.no_cache,
    )
