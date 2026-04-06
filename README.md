# NEUROGUARD
### Zero-Day IoT Compromise Detection via Siamese Behavioral Fingerprinting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

NEUROGUARD is a behavioral anomaly detection system for IoT devices that detects zero-day attacks **without ever training on attack data**. Instead of matching traffic against known attack signatures, it learns what each device's *normal* behavior looks like and alerts when a device deviates from its own established baseline.

> **Core claim:** A Siamese neural network trained only on normal IoT traffic can detect attacks it has never seen — because it learns what "normal" looks like for each device, and flags any deviation from that baseline.

This approach detects Mirai, DDoS, port scans, ransomware, backdoors, and any future attack type — by definition — because it never looks at attack signatures.

---

## Key Results (TON_IoT Benchmark)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9402** |
| Detection Rate | **97.6 – 98.3%** |
| False Positive Rate | 3.2% (1.8% with ≥10 enrollment windows) |
| F1 Score | **0.9814** at k=3.0σ |
| Backdoor detection | 98.74% |
| Ransomware detection | 97.50% |
| Embedding separation ratio | **3.08×** (inter/intra class) |

Evaluated on 839 real attack windows (8 attack types) and 347 normal windows — **zero attack data used during training, scaling, or enrollment.**

---

## Architecture

```
Raw network traffic (Zeek connection logs)
        ↓
Feature Extraction — 60-dimensional behavioral vector per 50-flow window
        ↓
Siamese Encoder — shared weights, contrastive loss (margin=3.0)
  Linear(60→128) + BN + ReLU
  Linear(128→256) + BN + ReLU + Dropout(0.3)
  Pre-LN TransformerEncoder (d=256, heads=4, layers=1)
  Linear(256→128) + ReLU → Linear(128→64)
        ↓
64-dim L2-normalized behavioral embedding
        ↓
Per-device DNA enrollment (centroid + threshold calibration)
        ↓
Cosine distance scoring → anomaly score → ALERT if score ≥ 1.0
        ↓
EWMA drift detector — catches slow APT-style compromise
```

**609,472 trainable parameters.** Runs on Apple Silicon (MPS), CUDA, or CPU.

---

## Project Structure

```
neuroguard/
├── src/
│   ├── features/
│   │   └── extractor.py        # 60-feature extraction from Zeek logs
│   ├── models/
│   │   ├── encoder.py          # BehavioralEncoder (Siamese shared weights)
│   │   ├── siamese.py          # SiameseNetwork + ContrastiveLoss
│   │   └── transformer.py      # Pre-LN TemporalEncoder block
│   ├── training/
│   │   ├── dataset.py          # WindowDataset + PairDataset
│   │   ├── train.py            # Training loop (AdamW + CosineAnnealing)
│   │   └── metrics.py          # ROC-AUC evaluation + per-attack-type breakdown
│   └── detection/
│       ├── enroll.py           # DeviceDNA enrollment
│       ├── scorer.py           # Real-time anomaly scoring + gradient attribution
│       └── drift.py            # EWMA behavioral drift detector
├── scripts/
│   ├── run_evaluation.py       # Full ROC-AUC evaluation pipeline
│   ├── run_baselines.py        # IsolationForest / OneClassSVM / Autoencoder comparison
│   └── live_demo_server.py     # FastAPI SSE server for live dashboard demo
├── dashboard/
│   └── neuroguard_demo.html    # Self-contained live demo dashboard
├── paper/
│   └── neuroguard_ieee.tex     # IEEE TIFS submission (LaTeX)
├── tests/                      # pytest test suite (78 tests)
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

# 3. Build feature windows and train
python -c "from src.training.dataset import build_windows; build_windows('data/raw/ton_iot/train_test_network.csv')"
python -c "from src.training.train import train; train()"

# 4. Run full evaluation
python scripts/run_evaluation.py

# 5. Launch live demo dashboard
python scripts/live_demo_server.py
# → Open dashboard/neuroguard_demo.html in browser
# → Click "Connect to Live Model"
```

---

## How It Works

### 1. Behavioral Fingerprinting (Training)
The Siamese network is trained on **pairs of 50-flow traffic windows**, labeled:
- **Positive pair (y=0):** both windows from the same device → should produce nearby embeddings
- **Negative pair (y=1):** windows from different devices → should produce distant embeddings

Contrastive loss: `L = (1-y)·d² + y·max(0, m-d)²` with margin m=3.0

Training uses **539,612 candidate pairs** from 1,360 normal windows across 16 IoT devices. Attack data is never seen.

### 2. Device DNA Enrollment
After training, each device is enrolled using held-out normal windows. A **per-device centroid** is computed in embedding space, and an alert threshold is calibrated at k=2.5–3.0 standard deviations above the enrollment mean cosine distance.

### 3. Real-Time Scoring
At inference, a new traffic window is embedded and its cosine distance to the device centroid is computed:

```
anomaly_score = cosine_distance / threshold
score ≥ 1.0  →  ALERT
```

Top contributing features are identified via gradient attribution (`∂score/∂input`).

### 4. EWMA Drift Detection
For slow APT-style compromise that stays below the per-window threshold, an EWMA detector (`α=0.1`) fires when the 20-window rolling mean exceeds 1.5× the enrollment baseline — catching long-dwell adversaries.

---

## Comparison with Baselines

All baselines trained on identical data (same 1,360 normal windows, same scaler):

| Method | ROC-AUC | Detection Rate | F1 |
|--------|---------|---------------|----|
| IsolationForest | 0.737 | 3.0% | 0.057 |
| OneClassSVM | 0.636 | 1.9% | 0.037 |
| Autoencoder | 0.267† | 2.0% | 0.039 |
| **NEUROGUARD (k=3.0)** | **0.940** | **97.6%** | **0.981** |

†Autoencoder exhibits inverted scoring on this dataset (reconstructs attack windows better than normal — a known failure mode). See paper §4.3 for analysis.

---

## Dataset

**TON_IoT** — Moustafa et al., UNSW Sydney (2020)
- 461,043 Zeek connection-log flow records
- 9 IoT device types across 16 IP addresses
- 8 attack types: backdoor, DDoS, DoS, injection, MITM, password, ransomware, scanning

Download: [https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)

---

## Paper

Full paper submitted to **IEEE Transactions on Information Forensics and Security (TIFS)**.

Source: `paper/neuroguard_ieee.tex` (IEEEtran format, compile with `pdflatex`)

---

## Tests

```bash
pytest tests/ -v   # 78 tests across all modules
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Author

**Sagar Bhetuwal**
