"""
NEUROGUARD — EWMA slow-compromise / behavioral drift detector.

Purpose
-------
The Siamese scorer catches sudden, large deviations (Mirai, ransomware,
DDoS) within 1–2 windows. But APT (Advanced Persistent Threat) actors
deliberately move slowly — each individual window may score below the
alert threshold while the device's behavior drifts steadily over hours
or days. This module catches that pattern.

Algorithm
---------
For each device, we maintain:

  ewma_t = alpha * score_t + (1 - alpha) * ewma_{t-1}

where score_t is the raw cosine distance from scorer.py (NOT the
normalized anomaly_score — we want the physical distance, not the
threshold-relative score, so the baseline comparison is meaningful).

Drift is flagged when, over a rolling window of W=20 observations:

  mean(ewma over last W windows) > drift_multiplier * baseline_mean

where:
  baseline_mean     = mean of raw distances during enrollment
                      (stored in dna.embedding_distances)
  drift_multiplier  = 1.5  (50% elevation above enrollment baseline)
  alpha             = 0.1  (slow-moving average — reacts over ~10 windows)
  W (window_size)   = 20   (minimum history before drift can fire)

Why these parameters?
---------------------
alpha=0.1: EWMA time constant = 1/alpha = 10 windows. With 50-flow
windows at typical IoT traffic rates, this covers several minutes of
activity. Slow enough to ignore transient spikes, fast enough to catch
sustained drift within 20–30 windows.

drift_multiplier=1.5: A 50% sustained elevation above the enrollment
baseline is statistically significant under typical IoT traffic
variance. In our dataset, enrollment baselines range from 0.0001 to
0.31 raw distance. A 1.5× multiplier means:
  - For tight devices (.184, baseline ~0.0001): fires at 0.00015 —
    this is actually useful, any deviation is significant
  - For variable devices (.152, baseline ~0.17): fires at 0.25 —
    requires sustained moderate anomaly, ignores transients

window_size=20: Requires 20 consecutive EWMA values to be evaluated.
Below this the rolling mean is statistically noisy. 20 windows × 50
flows = 1000 flows of sustained behavioral change must be observed
before a drift alert fires. This makes the detector very resistant to
false positives from legitimate device behavior changes (e.g. firmware
updates, scheduled maintenance).

Relationship to per-window scorer
----------------------------------
                Scorer (scorer.py)           Drift (drift.py)
Speed:          Immediate (1 window)         Slow (20+ windows)
Sensitivity:    High (catches large spikes)  Low (catches slow trends)
Target threat:  Ransomware, DDoS, Mirai      APT, C2 beaconing, slow exfil
FPR:            ~3–6%                        Near-zero (sustained change needed)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.detection.enroll import DeviceDNA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ALPHA:      float = 0.1
DEFAULT_MULTIPLIER: float = 1.5
DEFAULT_WINDOW:     int   = 20


# ---------------------------------------------------------------------------
# DriftAlert dataclass
# ---------------------------------------------------------------------------

@dataclass
class DriftAlert:
    """Emitted when sustained behavioral drift is detected.

    Attributes:
        device_id:       Device that drifted.
        trigger_window:  Index of the window that caused the alert (0-based).
        ewma_value:      EWMA value at trigger point.
        rolling_mean:    Mean of last window_size EWMA values.
        baseline_mean:   DNA enrollment baseline mean distance.
        drift_ratio:     rolling_mean / baseline_mean (e.g. 1.73 = 73% above baseline).
        timestamp:       UTC time of alert.
    """
    device_id:     str
    trigger_window: int
    ewma_value:    float
    rolling_mean:  float
    baseline_mean: float
    drift_ratio:   float
    timestamp:     datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"DriftAlert(device={self.device_id!r}, "
            f"window={self.trigger_window}, "
            f"drift_ratio={self.drift_ratio:.2f}x, "
            f"rolling_mean={self.rolling_mean:.4f}, "
            f"baseline={self.baseline_mean:.4f})"
        )


# ---------------------------------------------------------------------------
# EWMADriftDetector
# ---------------------------------------------------------------------------

class EWMADriftDetector:
    """Per-device EWMA drift detector.

    One instance per device. Feed raw cosine distances (from AnomalyResult
    .raw_distance) one at a time via update(). The detector maintains its
    own rolling EWMA history and fires DriftAlert when drift is confirmed.

    Args:
        device_id:        Device identifier.
        dna:              Enrolled DeviceDNA (provides baseline_mean).
        alpha:            EWMA smoothing factor (default: 0.1).
        drift_multiplier: Alert threshold multiplier over baseline (default: 1.5).
        window_size:      Rolling window length for mean comparison (default: 20).

    Example:
        >>> detector = EWMADriftDetector('192.168.1.193', dna)
        >>> for score in stream_of_raw_distances:
        ...     alert = detector.update(score)
        ...     if alert:
        ...         print(f"DRIFT DETECTED: {alert}")
    """

    def __init__(
        self,
        device_id:        str,
        dna:              DeviceDNA,
        alpha:            float = DEFAULT_ALPHA,
        drift_multiplier: float = DEFAULT_MULTIPLIER,
        window_size:      int   = DEFAULT_WINDOW,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if drift_multiplier <= 1.0:
            raise ValueError(f"drift_multiplier must be > 1.0, got {drift_multiplier}")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        self.device_id        = device_id
        self.alpha            = alpha
        self.drift_multiplier = drift_multiplier
        self.window_size      = window_size

        # Baseline mean: mean of enrollment cosine distances
        # dna.embedding_distances holds per-window distances from centroid
        if len(dna.embedding_distances) > 0:
            self.baseline_mean = float(dna.embedding_distances.mean())
        else:
            self.baseline_mean = 0.01  # fallback for 1-window enrollment

        # If baseline is effectively 0 (very tight device), use a small floor
        # to avoid division-by-zero and hair-trigger alerts
        self.baseline_mean = max(self.baseline_mean, 1e-4)

        self._drift_threshold = self.baseline_mean * self.drift_multiplier

        # State
        self._ewma: Optional[float] = None          # None until first update
        self._ewma_history: deque[float] = deque(maxlen=window_size)
        self._n_updates: int = 0
        self._alert_fired: bool = False             # fire once per drift episode

        logger.debug(
            f"DriftDetector({device_id}): baseline={self.baseline_mean:.4f} "
            f"threshold={self._drift_threshold:.4f} "
            f"alpha={alpha} window={window_size}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_distance: float) -> Optional[DriftAlert]:
        """Feed one new raw cosine distance observation.

        Args:
            raw_distance: Cosine distance from AnomalyResult.raw_distance.
                          Must be non-negative.

        Returns:
            DriftAlert if drift is confirmed this update, else None.
        """
        if raw_distance < 0:
            raise ValueError(f"raw_distance must be >= 0, got {raw_distance}")

        # Update EWMA
        if self._ewma is None:
            self._ewma = raw_distance   # initialize to first observation
        else:
            self._ewma = self.alpha * raw_distance + (1.0 - self.alpha) * self._ewma

        self._ewma_history.append(self._ewma)
        self._n_updates += 1

        # Need at least window_size observations to evaluate drift
        if len(self._ewma_history) < self.window_size:
            return None

        rolling_mean = float(np.mean(self._ewma_history))

        if rolling_mean > self._drift_threshold:
            if not self._alert_fired:
                self._alert_fired = True
                alert = DriftAlert(
                    device_id=self.device_id,
                    trigger_window=self._n_updates - 1,
                    ewma_value=self._ewma,
                    rolling_mean=rolling_mean,
                    baseline_mean=self.baseline_mean,
                    drift_ratio=rolling_mean / self.baseline_mean,
                )
                logger.warning(
                    f"DRIFT ALERT — {self.device_id}: "
                    f"rolling_mean={rolling_mean:.4f} > "
                    f"threshold={self._drift_threshold:.4f} "
                    f"({alert.drift_ratio:.2f}x baseline) "
                    f"at window {alert.trigger_window}"
                )
                return alert
        else:
            # Reset alert flag if drift subsides (allows re-alerting)
            self._alert_fired = False

        return None

    def reset(self) -> None:
        """Clear all state — call after a device is re-enrolled."""
        self._ewma = None
        self._ewma_history.clear()
        self._n_updates = 0
        self._alert_fired = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_ewma(self) -> Optional[float]:
        """Current EWMA value, or None if no updates yet."""
        return self._ewma

    @property
    def n_updates(self) -> int:
        """Total number of windows observed."""
        return self._n_updates

    @property
    def drift_threshold(self) -> float:
        """The raw-distance threshold above which rolling_mean triggers drift."""
        return self._drift_threshold

    @property
    def history(self) -> list[float]:
        """Copy of the current EWMA rolling window (up to window_size values)."""
        return list(self._ewma_history)

    def summary(self) -> dict:
        """Return a JSON-serializable summary of current detector state."""
        rolling_mean = float(np.mean(self._ewma_history)) if self._ewma_history else 0.0
        return {
            "device_id":      self.device_id,
            "n_updates":      self._n_updates,
            "current_ewma":   self._ewma,
            "rolling_mean":   rolling_mean,
            "baseline_mean":  self.baseline_mean,
            "drift_ratio":    rolling_mean / self.baseline_mean if self.baseline_mean > 0 else 0.0,
            "drift_threshold": self._drift_threshold,
            "alert_active":   self._alert_fired,
        }
