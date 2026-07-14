# NEUROGUARD
### Zero-Day IoT Compromise Detection via Siamese Behavioral Fingerprinting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

NEUROGUARD detects zero-day IoT attacks **without ever training on attack data**. Instead of matching traffic against known signatures, it learns what each device's *normal* behavior looks like and alerts when a device deviates from its own baseline.

> **Core idea:** A Siamese neural network trained only on normal IoT traffic can detect attacks it has never seen — because it learns what "normal" looks like per device, and flags any deviation from that baseline.

This means Mirai, DDoS, ransomware, backdoors, port scans, and any future attack type are detectable by definition — the model never needs to know what an attack looks like.

---

## Results (TON_IoT Benchmark)

Evaluated on a strict chronological 80/14/6 split — the model trains on the earliest 80% of normal traffic per device, enrolls on the middle 14%, and is tested on the final 6% (normal) plus all attack windows. **Zero attack data is used during training, scaling, or enrollment.**

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9875** |
| Detection Rate (TPR) | **99.64%** |
| False Positive Rate | **0.92%** (1/109 held-out normal windows) |
| F1 Score | **0.9976** at k=2.5σ |
| Embedding separation ratio | **72.08×** (inter/intra-class distance) |
| Intra-class distance | 0.060 |
| Inter-class distance | 4.297 |

### Multi-Seed Robustness (5 independent runs)

| Seed | AUC | TPR | FPR | F1 |
|------|-----|-----|-----|----|
| 42 | 0.9992 | 99.76% | 1.83% | 0.9976 |
| 1  | 0.9996 | 99.76% | 2.75% | 0.9970 |
| 2  | 0.9983 | 99.76% | 0.92% | 0.9982 |
| 3  | 0.9995 | 99.76% | 3.67% | 0.9964 |
| 4  | 0.9995 | 99.76% | 0.92% | 0.9982 |
| **Mean ± Std** | **0.9992 ± 0.0005** | **99.76% ± 0.00%** | 2.02% ± 1.07% | **0.9975 ± 0.0007** |

TPR is identically 99.76% across all five seeds — the detection rate is a structural property of the method, not a lucky initialization.

---

## Architecture

```
Raw network traffic (Zeek connection logs)
        ↓
Feature Extraction — 60-dimensional behavioral vector per 50-flow window
        ↓
Siamese Encoder (shared weights, contrastive loss margin=3.0)
  Linear(60→128) + BatchNorm + ReLU
  Linear(128→256) + BatchNorm + ReLU + Dropout(0.3)
  Pre-LN TransformerEncoder (d=256, heads=4, layers=1)
  Linear(256→128) + ReLU → Linear(128→64)
        ↓
64-dim L2-normalized behavioral embedding
        ↓
Per-device DNA enrollment (centroid + cosine threshold at mean + 2.5σ)
        ↓
Cosine distance scoring → ALERT if distance > threshold
        ↓
EWMA drift detector — catches slow APT-style compromise
```

**609,472 trainable parameters.** Runs on Apple Silicon (MPS), CUDA, or CPU.

---

## Comparison with Baselines

Each baseline receives the same per-device treatment as NEUROGUARD: trained on that device's own normal windows, evaluated on the same 109-window normal set and the same attack windows. Evaluated under the identical strict chronological protocol.

| Method | ROC-AUC | TPR | FPR |
|--------|---------|-----|-----|
| IsolationForest | 0.474† | — | — |
| OneClassSVM | — | — | — |
| Autoencoder | — | — | — |
| **NEUROGUARD** | **0.9875** | **99.64%** | **0.92%** |

†IsolationForest AUC drops from 0.974 (random split) to 0.474 under the chronological protocol, exposing temporal non-stationarity in TON_IoT traffic that random-split evaluation conceals.

---

## Ablations

| Variant | Sep Ratio | AUC | TPR | FPR |
|---------|-----------|-----|-----|-----|
| Full model (margin=3.0) | 72.08× | 0.9875 | 99.64% | 0.92% |
| A1: No Transformer | 12.34× | 0.9410 | 99.76% | 7.34% |
| A2: margin=2.0 | 17.26× | 0.9934 | 99.76% | 0.00% |

The Transformer block is responsible for the 5.8× improvement in embedding separation ratio (12.34× → 72.08×) and a 6.4 pp reduction in FPR. margin=3.0 is preferred over margin=2.0 for geometric robustness (72.08× vs 17.26× separation), though both achieve near-identical detection rates.

---

## Data Split Protocol

Normal windows per device are split **chronologically** by flow order:

```
First 80%  → train_normal   (Siamese training pairs)
Next  14%  → enroll_normal  (DNA centroid calibration only)
Last   6%  → test_normal    (FPR evaluation only — never touches threshold)
```

The chronological ordering prevents near-duplicate leakage from the 50%-stride overlapping windows. The last 10% of each device's `train_normal` is carved out as an internal validation set for early stopping, keeping `enroll_normal` strictly reserved for enrollment.

---

## Project Structure

```
neuroguard/
├── src/
│   ├── features/
│   │   └── extractor.py            # 60-feature extraction from Zeek logs
│   ├── models/
│   │   ├── encoder.py              # BehavioralEncoder (Siamese shared weights)
│   │   ├── siamese.py              # SiameseNetwork + ContrastiveLoss
│   │   └── transformer.py          # Pre-LN TemporalEncoder block
│   ├── training/
│   │   ├── dataset.py              # WindowDataset + PairDataset (chronological split)
│   │   ├── train.py                # Training loop (AdamW + CosineAnnealing)
│   │   └── metrics.py              # ROC-AUC evaluation
│   └── detection/
│       ├── enroll.py               # DeviceDNA enrollment
│       ├── scorer.py               # Real-time anomaly scoring
│       └── drift.py                # EWMA behavioral drift detector
├── scripts/
│   ├── generate_roc_comparison.py  # ROC curves: NEUROGUARD vs baselines
│   ├── run_baselines_per_device.py # Per-device baseline evaluation
│   ├── multi_seed_eval.py          # 5-seed robustness evaluation
│   ├── centroid_stability.py       # Bootstrap centroid stability experiment
│   ├── run_iot23_eval.py           # Cross-dataset evaluation on IoT-23
│   └── run_zero_shot_iot23.py      # Zero-shot IoT-23 evaluation
├── ablation_a1_no_transformer.py   # Ablation: encoder without Transformer
├── ablation_a2_margin2.py          # Ablation: contrastive margin=2.0
├── models/checkpoints/             # Saved model weights + scalers
├── tests/                          # pytest test suite
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/alphago004/Neuroguard.git
cd Neuroguard
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download TON_IoT dataset
# → https://research.unsw.edu.au/projects/toniot-datasets
# → Place train_test_network.csv in data/raw/ton_iot/

# 3. Train
python -m src.training.train

# 4. Run full evaluation
python scripts/generate_roc_comparison.py

# 5. Run ablations
python ablation_a1_no_transformer.py
python ablation_a2_margin2.py
```

---

## Dataset

**TON_IoT** — Moustafa et al., UNSW Sydney (2020)
- 461,043 Zeek connection-log flow records
- 9 IoT device types across 16 IP addresses
- 8 attack types: backdoor, DDoS, DoS, injection, MITM, password, ransomware, scanning

Download: [https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Author

**Sagar Bhetuwal**
