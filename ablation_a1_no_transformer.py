"""
Ablation A1: No Transformer
Identical to main training but with the TemporalEncoder removed from the encoder.
Run with: python3 ablation_a1_no_transformer.py
Results saved to: ablation_a1_results.txt
"""

import sys
import pickle
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from loguru import logger

from src.training.dataset import WindowDataset, PairDataset, WindowRecord, build_windows
from src.models.siamese import SiameseNetwork, ContrastiveLoss
from src.models.encoder import BehavioralEncoder
from src.detection.enroll import enroll_all_devices
from src.training.metrics import evaluate_model

TON_IOT_CSV  = PROJECT_ROOT / "data" / "raw" / "ton_iot" / "train_test_network.csv"
SAVE_DIR     = PROJECT_ROOT / "models" / "checkpoints"
WINDOW_CACHE = PROJECT_ROOT / "data" / "processed" / "window_dataset.pkl"

EPOCHS     = 100
BATCH_SIZE = 128
LR         = 1e-3
PATIENCE   = 100
DEVICE_CAP = 50
N_PAIRS    = 100_000
MARGIN     = 3.0


class BehavioralEncoderNoTransformer(BehavioralEncoder):
    """BehavioralEncoder with TemporalEncoder bypassed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)      # (batch, 128)
        x = self.layer2(x)      # (batch, 256)
        # self.temporal_encoder skipped
        x = self.projection(x)  # (batch, 64)
        return x


def build_model_no_transformer():
    encoder = BehavioralEncoderNoTransformer()
    model   = SiameseNetwork(encoder)
    loss_fn = ContrastiveLoss(margin=MARGIN)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"A1 model (no Transformer): {n_params:,} trainable parameters")
    return model, loss_fn


def fit_scaler(records):
    X = np.stack([r.features for r in records])
    scaler = RobustScaler()
    scaler.fit(X)
    logger.info(f"RobustScaler fitted on {len(records)} train_normal windows")
    return scaler


def apply_scaler(records, scaler):
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


@torch.no_grad()
def compute_distances(model, records, device, batch_size=256):
    model.eval()
    all_features   = np.stack([r.features for r in records])
    all_device_ids = [r.device_id for r in records]

    embeddings = []
    for start in range(0, len(all_features), batch_size):
        batch = torch.from_numpy(all_features[start:start + batch_size]).to(device)
        emb   = model.encoder(batch)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    device_to_indices: dict = {}
    for i, dev in enumerate(all_device_ids):
        device_to_indices.setdefault(dev, []).append(i)
    devices = list(device_to_indices.keys())

    rng = np.random.default_rng(42)
    intra_dists = []
    for dev, indices in device_to_indices.items():
        if len(indices) < 2:
            continue
        indices = np.array(indices)
        n_pairs = min(500, len(indices) * (len(indices) - 1) // 2)
        i_idx   = rng.choice(indices, n_pairs)
        j_idx   = rng.choice(indices, n_pairs)
        mask    = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        if len(i_idx) == 0:
            continue
        dists = torch.norm(embeddings[i_idx] - embeddings[j_idx], dim=1)
        intra_dists.extend(dists.tolist())

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
    return {"intra_mean": intra_mean, "inter_mean": inter_mean, "separation_ratio": ratio}


def main():
    t_start = time.time()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"=== ABLATION A1: NO TRANSFORMER === device={device}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    if WINDOW_CACHE.exists():
        logger.info(f"Loading WindowDataset cache: {WINDOW_CACHE}")
        window_ds = WindowDataset.load(WINDOW_CACHE)
    else:
        logger.info("Building WindowDataset from CSV (takes ~30s on Pi)...")
        window_ds = build_windows(TON_IOT_CSV)
        window_ds.save(WINDOW_CACHE)

    stats = window_ds.stats()
    logger.info(f"Dataset stats: {stats}")

    scaler = fit_scaler(window_ds.train_normal)
    with open(SAVE_DIR / "scaler_a1.pkl", "wb") as f:
        pickle.dump(scaler, f)

    scaled_train  = apply_scaler(window_ds.train_normal,  scaler)
    scaled_enroll = apply_scaler(window_ds.enroll_normal, scaler)
    scaled_test   = apply_scaler(window_ds.test_normal,   scaler)

    # Carve internal val from last 10% of train_normal per device (chronological)
    per_dev = {}
    for r in scaled_train:
        per_dev.setdefault(r.device_id, []).append(r)
    actual_train, internal_val = [], []
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
    scaled_ds.attack_records = apply_scaler(window_ds.attack_records, scaler)
    scaled_ds.device_ids     = window_ds.device_ids
    scaled_ds.device_to_idx  = window_ds.device_to_idx
    scaled_ds._seed          = window_ds._seed

    val_pairs  = PairDataset(scaled_ds, split="enroll", n_pairs=N_PAIRS, device_cap=DEVICE_CAP, seed=0)
    val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, loss_fn = build_model_no_transformer()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improv = 0

    logger.info(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Status'}")
    logger.info("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        epoch_pairs = PairDataset(
            scaled_ds, split="train",
            n_pairs=N_PAIRS, device_cap=DEVICE_CAP, seed=epoch * 7,
        )
        train_loader = DataLoader(epoch_pairs, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

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
        scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improv = 0
            torch.save(model.state_dict(), SAVE_DIR / "ablation_a1_no_transformer.pt")
            status = "saved"
        else:
            epochs_no_improv += 1
            status = f"no improv {epochs_no_improv}/{PATIENCE}"

        logger.info(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  {status}")

        if epochs_no_improv >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    a1_checkpoint = SAVE_DIR / "ablation_a1_no_transformer.pt"
    best_model, _ = build_model_no_transformer()
    best_model.load_state_dict(
        torch.load(a1_checkpoint, map_location=device)
    )
    best_model = best_model.to(device)
    dist_metrics = compute_distances(best_model, scaled_ds.test_normal, device)

    # ── Full evaluation: enroll + score + ROC-AUC (using unscaled window_ds) ──
    logger.info("Running full evaluation on A1 checkpoint…")
    dna_map = enroll_all_devices(window_ds, scaler, checkpoint_path=a1_checkpoint)
    report  = evaluate_model(window_ds, dna_map, scaler, checkpoint_path=a1_checkpoint)

    roc_auc = report.roc_auc
    tpr_25  = report.confusion_2_5.tpr
    fpr_25  = report.confusion_2_5.fpr
    f1_25   = report.confusion_2_5.f1
    tpr_30  = report.confusion_3_0.tpr
    fpr_30  = report.confusion_3_0.fpr
    f1_30   = report.confusion_3_0.f1

    total_time = time.time() - t_start

    logger.info("\n" + "=" * 55)
    logger.info("ABLATION A1 (NO TRANSFORMER) COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Best epoch:        {best_epoch}")
    logger.info(f"  Best val loss:     {best_val_loss:.6f}")
    logger.info(f"  Intra-class dist:  {dist_metrics['intra_mean']:.4f}  (target < 0.5)")
    logger.info(f"  Inter-class dist:  {dist_metrics['inter_mean']:.4f}  (target > 1.5)")
    logger.info(f"  Separation ratio:  {dist_metrics['separation_ratio']:.2f}x")
    logger.info(f"  ROC-AUC:           {roc_auc:.4f}")
    logger.info(f"  TPR (k=2.5):       {tpr_25:.4f}")
    logger.info(f"  FPR (k=2.5):       {fpr_25:.4f}")
    logger.info(f"  F1  (k=2.5):       {f1_25:.4f}")
    logger.info(f"  TPR (k=3.0):       {tpr_30:.4f}")
    logger.info(f"  F1  (k=3.0):       {f1_30:.4f}")
    logger.info(f"  Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 55)

    with open(PROJECT_ROOT / "ablation_a1_results.txt", "w") as f:
        f.write("ABLATION A1 - NO TRANSFORMER\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"intra_mean: {dist_metrics['intra_mean']:.4f}\n")
        f.write(f"inter_mean: {dist_metrics['inter_mean']:.4f}\n")
        f.write(f"separation_ratio: {dist_metrics['separation_ratio']:.4f}\n")
        f.write(f"roc_auc: {roc_auc:.4f}\n")
        f.write(f"tpr_25: {tpr_25:.4f}\n")
        f.write(f"fpr_25: {fpr_25:.4f}\n")
        f.write(f"f1_25: {f1_25:.4f}\n")
        f.write(f"tpr_30: {tpr_30:.4f}\n")
        f.write(f"fpr_30: {fpr_30:.4f}\n")
        f.write(f"f1_30: {f1_30:.4f}\n")
        f.write(f"total_time_s: {total_time:.1f}\n")
    logger.info("Results saved → ablation_a1_results.txt")


if __name__ == "__main__":
    main()
