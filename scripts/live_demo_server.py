"""
NEUROGUARD — Live Demo Server
==============================
Minimal FastAPI server that streams REAL anomaly scores from the trained
Siamese model to the dashboard via Server-Sent Events (SSE).

What is "real" here:
  - Loads best_model.pt (trained Siamese encoder, 609,472 params)
  - Loads enrolled DeviceDNA for 192.168.1.193 (Thermostat)
  - Normal mode  : cycles through the 16 real test_normal windows
  - Attack mode  : cycles through real TON_IoT attack windows (backdoor /
                   ransomware / DoS recorded on that device)
  - score_window() computes cosine distance → actual anomaly score
  - Scores stream to dashboard every 2 seconds via SSE

Usage:
    cd neuroGuard
    source venv/bin/activate
    python scripts/live_demo_server.py
    Then open dashboard/neuroguard_demo.html and click "Connect to Live Model"
"""

import asyncio
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
CHECKPOINT  = ROOT / "models" / "checkpoints" / "best_model.pt"
SCALER_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"
CACHE_PATH  = ROOT / "data" / "processed" / "window_dataset.pkl"
DNA_DIR     = ROOT / "data" / "processed" / "dna"

sys.path.insert(0, str(ROOT))

from src.training.dataset import WindowDataset
from src.detection.scorer import score_window
from src.detection.enroll import DeviceDNA

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="NEUROGUARD Live Demo", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
attack_mode   : bool = False
window_ds     = None
scaler        = None
dna_map       : dict = {}
normal_windows: list = []   # scaled test_normal WindowRecords for .193
attack_windows: list = []   # scaled attack WindowRecords for .193
normal_idx    : int  = 0
attack_idx    : int  = 0

THERMOSTAT_IP = "192.168.1.193"
INTERVAL_S    = 2.0   # seconds between score updates

# ── Startup: load everything once ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global window_ds, scaler, dna_map, normal_windows, attack_windows

    logger.info("Loading WindowDataset…")
    window_ds = WindowDataset.load(CACHE_PATH)

    logger.info("Loading scaler…")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Loading enrolled DNA…")
    for pkl_path in DNA_DIR.glob("*.pkl"):
        dev_id = pkl_path.stem.replace("_", ".")
        with open(pkl_path, "rb") as f:
            dna_map[dev_id] = pickle.load(f)

    # Scale the windows we'll use (same transform as full evaluation)
    def scale_records(records):
        if not records:
            return []
        import numpy as np
        from src.training.dataset import WindowRecord
        X = np.stack([r.features for r in records]).astype(np.float32)
        Xs = scaler.transform(X).astype(np.float32)
        return [
            WindowRecord(r.device_id, Xs[i], r.label, r.window_idx, r.flow_start)
            for i, r in enumerate(records)
        ]

    normal_raw = [r for r in window_ds.test_normal if r.device_id == THERMOSTAT_IP]
    attack_raw = [r for r in window_ds.attack_records if r.device_id == THERMOSTAT_IP]

    normal_windows = scale_records(normal_raw)
    attack_windows = scale_records(attack_raw)

    logger.success(
        f"Ready — {len(normal_windows)} normal windows, "
        f"{len(attack_windows)} attack windows for {THERMOSTAT_IP}"
    )
    logger.info("Dashboard: open dashboard/neuroguard_demo.html → click 'Connect to Live Model'")


# ── /trigger-attack ────────────────────────────────────────────────────────────
@app.post("/trigger-attack")
async def trigger_attack():
    global attack_mode, attack_idx
    attack_mode = True
    attack_idx  = 0
    logger.warning("Attack mode ACTIVATED — feeding real attack windows to model")
    return {"status": "attack_started"}


# ── /trigger-normal ────────────────────────────────────────────────────────────
@app.post("/trigger-normal")
async def trigger_normal():
    global attack_mode, normal_idx
    attack_mode = False
    normal_idx  = 0
    logger.info("Normal mode restored")
    return {"status": "normal_restored"}


# ── /status ────────────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    dna = dna_map.get(THERMOSTAT_IP)
    return {
        "model_loaded":     CHECKPOINT.exists(),
        "normal_windows":   len(normal_windows),
        "attack_windows":   len(attack_windows),
        "enrolled_devices": len(dna_map),
        "threshold":        round(dna.threshold_distance, 5) if dna else None,
        "baseline_mean":    round(float(dna.embedding_distances.mean()), 5) if dna else None,
        "attack_mode":      attack_mode,
    }


# ── /stream  (SSE) ─────────────────────────────────────────────────────────────
@app.get("/stream")
async def stream():
    """
    Server-Sent Events endpoint.
    Sends one JSON event every INTERVAL_S seconds:
      { score, raw_distance, threshold, mode, window_idx, timestamp }
    """
    async def event_generator():
        global normal_idx, attack_idx, attack_mode

        dna = dna_map.get(THERMOSTAT_IP)
        if dna is None:
            yield f"data: {json.dumps({'error': 'DNA not loaded'})}\n\n"
            return

        # Send initial status immediately
        status_data = {
            "type":          "init",
            "threshold":     round(dna.threshold_distance, 5),
            "baseline_mean": round(float(dna.embedding_distances.mean()), 5),
            "n_normal":      len(normal_windows),
            "n_attack":      len(attack_windows),
        }
        yield f"data: {json.dumps(status_data)}\n\n"

        while True:
            try:
                if attack_mode and attack_windows:
                    record    = attack_windows[attack_idx % len(attack_windows)]
                    attack_idx = (attack_idx + 1) % len(attack_windows)
                    mode_label = "attack"
                elif normal_windows:
                    record    = normal_windows[normal_idx % len(normal_windows)]
                    normal_idx = (normal_idx + 1) % len(normal_windows)
                    mode_label = "normal"
                else:
                    await asyncio.sleep(INTERVAL_S)
                    continue

                result = score_window(
                    device_id=THERMOSTAT_IP,
                    features=record.features,
                    dna=dna,
                    checkpoint_path=CHECKPOINT,
                )

                payload = {
                    "type":         "score",
                    "score":        round(result.anomaly_score, 5),
                    "raw_distance": round(result.raw_distance, 5),
                    "threshold":    round(result.threshold, 5),
                    "status":       result.status,
                    "mode":         mode_label,
                    "window_idx":   int(record.window_idx),
                    "timestamp":    time.strftime("%H:%M:%S"),
                }

                logger.debug(
                    f"[{mode_label.upper():6}] score={result.anomaly_score:.4f} "
                    f"dist={result.raw_distance:.4f} thr={result.threshold:.4f} "
                    f"→ {result.status}"
                )

                yield f"data: {json.dumps(payload)}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

            await asyncio.sleep(INTERVAL_S)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting NEUROGUARD Live Demo Server on http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
