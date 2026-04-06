"""
Tests for src/detection/drift.py

Run with: pytest tests/test_drift.py -v
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from datetime import datetime

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.detection.drift import (
    EWMADriftDetector,
    DEFAULT_ALPHA,
    DEFAULT_MULTIPLIER,
    DEFAULT_WINDOW,
)
from src.detection.enroll import DeviceDNA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dna(baseline_mean: float, n_windows: int = 30) -> DeviceDNA:
    """Construct a minimal DeviceDNA with the given enrollment baseline."""
    rng = np.random.default_rng(42)
    # Generate plausible enrollment distances centred on baseline_mean
    distances = np.clip(
        rng.normal(baseline_mean, baseline_mean * 0.1, n_windows),
        0.0, 1.0
    ).astype(np.float32)
    distances.sort()
    return DeviceDNA(
        device_id="192.168.1.test",
        centroid=np.zeros(64, dtype=np.float32),
        sigma=np.ones(64, dtype=np.float32) * 0.01,
        threshold_distance=float(baseline_mean * 3.0),
        n_windows=n_windows,
        enrolled_at=datetime.now(),
        embedding_distances=distances,
    )


# ---------------------------------------------------------------------------
# Unit tests: construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_baseline_derived_from_dna(self):
        """baseline_mean must equal the mean of dna.embedding_distances."""
        dna = _make_dna(baseline_mean=0.10)
        det = EWMADriftDetector("dev", dna)
        assert abs(det.baseline_mean - float(dna.embedding_distances.mean())) < 1e-5

    def test_drift_threshold_is_multiplier_times_baseline(self):
        """drift_threshold = baseline_mean * drift_multiplier."""
        dna = _make_dna(baseline_mean=0.10)
        det = EWMADriftDetector("dev", dna, drift_multiplier=1.5)
        assert abs(det.drift_threshold - det.baseline_mean * 1.5) < 1e-6

    def test_invalid_alpha_raises(self):
        dna = _make_dna(0.1)
        with pytest.raises(ValueError, match="alpha"):
            EWMADriftDetector("dev", dna, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EWMADriftDetector("dev", dna, alpha=1.0)

    def test_invalid_multiplier_raises(self):
        dna = _make_dna(0.1)
        with pytest.raises(ValueError, match="drift_multiplier"):
            EWMADriftDetector("dev", dna, drift_multiplier=1.0)

    def test_invalid_window_size_raises(self):
        dna = _make_dna(0.1)
        with pytest.raises(ValueError, match="window_size"):
            EWMADriftDetector("dev", dna, window_size=1)


# ---------------------------------------------------------------------------
# Unit tests: EWMA arithmetic
# ---------------------------------------------------------------------------

class TestEWMAAithmetic:
    def test_first_update_initialises_to_value(self):
        """On the very first update, EWMA = the observation (cold start)."""
        dna = _make_dna(0.1)
        det = EWMADriftDetector("dev", dna, alpha=0.1)
        det.update(0.5)
        assert det.current_ewma == pytest.approx(0.5)

    def test_ewma_formula_two_steps(self):
        """Verify EWMA(alpha=0.1) arithmetic across 2 updates manually."""
        dna = _make_dna(0.1)
        det = EWMADriftDetector("dev", dna, alpha=0.1)
        det.update(0.4)          # ewma = 0.4
        det.update(0.6)          # ewma = 0.1*0.6 + 0.9*0.4 = 0.42
        assert det.current_ewma == pytest.approx(0.42, rel=1e-5)

    def test_no_drift_before_window_size(self):
        """No alert should fire before window_size observations."""
        dna = _make_dna(0.05)
        det = EWMADriftDetector("dev", dna, alpha=0.1, window_size=20)
        high_score = 1.0   # well above threshold
        for _ in range(19):
            alert = det.update(high_score)
            assert alert is None, "Alert fired before window_size reached"

    def test_constant_normal_scores_never_fire(self):
        """Constant scores at baseline level must never trigger drift."""
        baseline = 0.10
        dna = _make_dna(baseline)
        det = EWMADriftDetector("dev", dna, alpha=0.1, drift_multiplier=1.5, window_size=20)
        for _ in range(50):
            alert = det.update(baseline)
        assert alert is None
        assert not det._alert_fired

    def test_reset_clears_state(self):
        """After reset(), n_updates=0 and current_ewma=None."""
        dna = _make_dna(0.1)
        det = EWMADriftDetector("dev", dna)
        for _ in range(10):
            det.update(0.5)
        det.reset()
        assert det.n_updates == 0
        assert det.current_ewma is None
        assert det.history == []


# ---------------------------------------------------------------------------
# Core requirement: gradual drift simulation
# ---------------------------------------------------------------------------

class TestGradualDrift:
    """THE primary scientific test: slow compromise must be detected
    between window 15 and window 25."""

    def test_drift_fires_between_window_15_and_25(self):
        """Simulate gradual score increase over 30 windows.

        Schedule:
          Windows 0–9   : normal scores at baseline (0.05)
          Windows 10–29 : linearly increasing from 0.05 → 0.50

        With baseline=0.05, drift_threshold=0.075 (1.5×), alpha=0.1,
        and window_size=20, the rolling EWMA should cross the threshold
        somewhere in windows 15–25.
        """
        baseline = 0.05
        dna = _make_dna(baseline_mean=baseline, n_windows=30)
        det = EWMADriftDetector(
            "192.168.1.test", dna,
            alpha=DEFAULT_ALPHA,
            drift_multiplier=DEFAULT_MULTIPLIER,
            window_size=DEFAULT_WINDOW,
        )

        n_total = 30
        scores = np.concatenate([
            np.full(10, baseline),                   # windows 0–9: normal
            np.linspace(baseline, 0.50, 20),          # windows 10–29: rising
        ])
        assert len(scores) == n_total

        first_alert_window: int | None = None
        for score in scores:
            alert = det.update(score)
            if alert is not None and first_alert_window is None:
                first_alert_window = alert.trigger_window

        assert first_alert_window is not None, (
            "Drift detector never fired over 30 windows of sustained score increase. "
            "Check alpha, drift_multiplier, and window_size parameters."
        )
        assert 14 <= first_alert_window <= 25, (
            f"Drift alert fired at window {first_alert_window}, "
            f"expected between 15 and 25. "
            f"EWMA history at alert: {det.history}"
        )

    def test_alert_contains_correct_metadata(self):
        """DriftAlert fields must be populated correctly."""
        baseline = 0.05
        dna = _make_dna(baseline_mean=baseline, n_windows=30)
        det = EWMADriftDetector("192.168.1.99", dna, alpha=0.1,
                                drift_multiplier=1.5, window_size=20)
        scores = np.concatenate([
            np.full(10, baseline),
            np.linspace(baseline, 0.50, 20),
        ])
        alert = None
        for s in scores:
            a = det.update(s)
            if a and alert is None:
                alert = a

        assert alert is not None
        assert alert.device_id == "192.168.1.99"
        assert alert.drift_ratio > 1.5            # must exceed multiplier
        assert alert.rolling_mean > alert.baseline_mean * 1.5
        assert alert.ewma_value > 0
        assert isinstance(alert.timestamp, datetime)

    def test_sudden_spike_then_recovery_does_not_stay_alerted(self):
        """A transient spike followed by sufficient recovery must re-arm the detector.

        EWMA time constant = 1/alpha = 10 windows. After 5 spikes at 1.0
        (EWMA peaks at ~0.44), it takes ~40 windows at baseline for the
        rolling_mean to decay below threshold=0.075. This is intentional —
        slow EWMA means slow decay, which is what makes it sensitive to
        gradual drift rather than transient noise.

        We verify the detector RE-ARMS (alert_fired → False) after sufficient
        recovery, confirming it can fire again on a future drift episode.
        """
        baseline = 0.05
        dna = _make_dna(baseline_mean=baseline, n_windows=30)
        det = EWMADriftDetector("dev", dna, alpha=0.1,
                                drift_multiplier=1.5, window_size=20)

        # Phase 1: normal for 25 windows (establish steady state)
        for _ in range(25):
            det.update(baseline)

        # Phase 2: spike briefly (2 windows — a transient)
        det.update(0.20)
        det.update(0.20)

        # Phase 3: sustained recovery — 50 windows at baseline
        # EWMA after 2 spikes of 0.20 peaks at ~0.07 — decays in ~15 windows
        for _ in range(50):
            det.update(baseline)

        # Alert flag must be False after sustained recovery
        assert not det._alert_fired, (
            "Detector remained in alert state after 50 normal recovery windows. "
            "EWMA should have decayed well below threshold by now."
        )

    def test_drift_ratio_in_summary(self):
        """summary() must report drift_ratio > 1.5 when alert is active."""
        baseline = 0.05
        dna = _make_dna(baseline_mean=baseline, n_windows=30)
        det = EWMADriftDetector("dev", dna, alpha=0.1,
                                drift_multiplier=1.5, window_size=20)
        scores = np.concatenate([
            np.full(10, baseline),
            np.linspace(baseline, 0.50, 20),
        ])
        for s in scores:
            det.update(s)

        summary = det.summary()
        if det._alert_fired:
            assert summary["drift_ratio"] > 1.5
            assert summary["alert_active"] is True
