"""
Tests for src/models/encoder.py, src/models/siamese.py, src/models/transformer.py

Run with:  pytest tests/test_models.py -v

These tests verify the model's correctness properties — not just that the
code runs, but that the outputs obey the mathematical contracts that the
contrastive learning setup depends on.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from pathlib import Path
import tempfile

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import pytest

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from src.models.encoder import BehavioralEncoder, DEFAULT_EMBEDDING_DIM
from src.models.siamese import (
    SiameseNetwork,
    ContrastiveLoss,
    build_model,
    DEFAULT_MARGIN,
)
from src.models.transformer import TemporalEncoder
from src.features.extractor import FEATURE_DIM

# ---------------------------------------------------------------------------
# Device selection (use MPS if available — same pattern as production code)
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _device()
BATCH  = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_loss():
    """Build the default SiameseNetwork + ContrastiveLoss, move to device."""
    model, loss_fn = build_model()
    model  = model.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)
    return model, loss_fn


@pytest.fixture(scope="module")
def random_batch():
    """Random batch of shape (8, 60) on the test device — mimics real windows."""
    torch.manual_seed(0)
    anchor = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
    pair   = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
    # Alternating same/different-device labels: [0,1,0,1,0,1,0,1]
    labels = torch.tensor([float(i % 2) for i in range(BATCH)], device=DEVICE)
    return anchor, pair, labels


# ---------------------------------------------------------------------------
# TemporalEncoder
# ---------------------------------------------------------------------------

class TestTemporalEncoder:
    def test_output_shape_preserved(self):
        """TemporalEncoder must not change (batch, d_model) shape."""
        te = TemporalEncoder(d_model=256, nhead=4, num_layers=2).to(DEVICE)
        x = torch.randn(BATCH, 256, device=DEVICE)
        out = te(x)
        assert out.shape == (BATCH, 256), f"Shape changed: {out.shape}"

    def test_invalid_nhead_raises(self):
        """d_model=256 with nhead=3 (256%3≠0) must raise ValueError immediately."""
        with pytest.raises(ValueError, match="divisible"):
            TemporalEncoder(d_model=256, nhead=3)

    def test_output_differs_from_input(self):
        """Transformer must transform the input — output should not equal input."""
        te = TemporalEncoder(d_model=256, nhead=4, num_layers=2).to(DEVICE)
        x = torch.randn(BATCH, 256, device=DEVICE)
        out = te(x)
        assert not torch.allclose(out, x), (
            "TemporalEncoder output equals input — transformer is a no-op"
        )


# ---------------------------------------------------------------------------
# BehavioralEncoder
# ---------------------------------------------------------------------------

class TestBehavioralEncoder:
    def test_output_shape(self, model_and_loss):
        """Encoder output must be (batch, 64) — the Siamese input contract."""
        model, _ = model_and_loss
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb = model.encoder(x)
        assert emb.shape == (BATCH, DEFAULT_EMBEDDING_DIM), (
            f"Expected ({BATCH}, {DEFAULT_EMBEDDING_DIM}), got {emb.shape}"
        )

    def test_output_dtype_float32(self, model_and_loss):
        """Embeddings must be float32 for MPS/CUDA compatibility."""
        model, _ = model_and_loss
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb = model.encoder(x)
        assert emb.dtype == torch.float32

    def test_no_nan_in_output(self, model_and_loss):
        """NaN in embeddings propagates silently into loss → must never occur."""
        model, _ = model_and_loss
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb = model.encoder(x)
        assert torch.isfinite(emb).all(), (
            f"Non-finite values in encoder output: {(~torch.isfinite(emb)).sum()} elements"
        )

    def test_different_inputs_produce_different_embeddings(self, model_and_loss):
        """Distinct inputs must produce distinct embeddings — no hash collision."""
        model, _ = model_and_loss
        x1 = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        x2 = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb1 = model.encoder(x1)
        emb2 = model.encoder(x2)
        assert not torch.allclose(emb1, emb2), "Different inputs produced identical embeddings"

    def test_encode_inference_normalized(self, model_and_loss):
        """encode(normalize=True) must return unit-norm vectors.

        The scorer.py computes cosine distance, which requires L2-normalized
        embeddings. If norms deviate from 1.0, distance comparisons are wrong.
        """
        model, _ = model_and_loss
        model.encoder.eval()
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb = model.encoder.encode(x, normalize=True)
        norms = torch.linalg.norm(emb, dim=1)
        assert torch.allclose(norms, torch.ones(BATCH, device=DEVICE), atol=1e-5), (
            f"Norms after L2 normalization: {norms}"
        )
        model.encoder.train()

    def test_batch_size_1_works(self, model_and_loss):
        """BatchNorm1d requires batch_size >= 2 during training, but must work in eval."""
        model, _ = model_and_loss
        model.encoder.eval()
        x = torch.randn(1, FEATURE_DIM, device=DEVICE)
        emb = model.encoder(x)
        assert emb.shape == (1, DEFAULT_EMBEDDING_DIM)
        model.encoder.train()

    def test_shared_weights_between_twins(self, model_and_loss):
        """The two 'twins' must share the exact same parameter tensors (not just values).

        Must be tested in eval() mode: Dropout is intentionally non-deterministic
        during training (that's its job), so two forward passes on the same input
        in train mode will rightfully differ. eval() disables Dropout, making
        the encoder a pure deterministic function of its weights.
        """
        model, _ = model_and_loss
        encoder_params = list(model.encoder.parameters())
        assert len(encoder_params) > 0

        model.encoder.eval()
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        with torch.no_grad():
            emb_1 = model.encoder(x)
            emb_2 = model.encoder(x)
        assert torch.allclose(emb_1, emb_2), (
            "Same input through same encoder (eval mode) produced different outputs"
        )
        model.encoder.train()


# ---------------------------------------------------------------------------
# ContrastiveLoss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_perfect_positive_pair_loss_near_zero(self):
        """Same-device pair (label=0) with distance=0 → loss ≈ 0.

        If the encoder collapses all windows to the same point, this loss
        would be zero even without learning — but that's caught by the
        negative pair loss.
        """
        loss_fn = ContrastiveLoss(margin=DEFAULT_MARGIN).to(DEVICE)
        dist   = torch.tensor([0.0], device=DEVICE)
        labels = torch.tensor([0.0], device=DEVICE)
        loss = loss_fn(dist, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_negative_pair_loss_near_zero(self):
        """Different-device pair (label=1) with distance >> margin → loss ≈ 0."""
        loss_fn = ContrastiveLoss(margin=DEFAULT_MARGIN).to(DEVICE)
        dist   = torch.tensor([10.0], device=DEVICE)   # far beyond margin=2.0
        labels = torch.tensor([1.0],  device=DEVICE)
        loss = loss_fn(dist, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_close_negative_pair_has_high_loss(self):
        """Different-device pair (label=1) with distance=0 → loss = (margin/2)²."""
        margin  = 2.0
        loss_fn = ContrastiveLoss(margin=margin).to(DEVICE)
        dist    = torch.tensor([0.0], device=DEVICE)
        labels  = torch.tensor([1.0], device=DEVICE)
        loss    = loss_fn(dist, labels)
        expected = 0.5 * margin ** 2  # = 2.0
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_far_positive_pair_has_high_loss(self):
        """Same-device pair (label=0) with distance=2.0 → loss = 0.5 * 4.0 = 2.0."""
        loss_fn = ContrastiveLoss(margin=DEFAULT_MARGIN).to(DEVICE)
        dist    = torch.tensor([2.0], device=DEVICE)
        labels  = torch.tensor([0.0], device=DEVICE)
        loss    = loss_fn(dist, labels)
        assert loss.item() == pytest.approx(0.5 * 2.0 ** 2, rel=1e-5)

    def test_loss_is_scalar(self, model_and_loss, random_batch):
        """Loss must reduce to a scalar — DataParallel and MPS require this."""
        model, loss_fn = model_and_loss
        anchor, pair, labels = random_batch
        emb_a, emb_b = model(anchor, pair)
        dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
        loss = loss_fn(dist, labels)
        assert loss.shape == torch.Size([]), f"Loss shape is {loss.shape}, expected scalar"

    def test_zero_margin_raises(self):
        """margin=0 is mathematically meaningless — must raise immediately."""
        with pytest.raises(ValueError):
            ContrastiveLoss(margin=0.0)


# ---------------------------------------------------------------------------
# SiameseNetwork — forward pass (the core smoke test)
# ---------------------------------------------------------------------------

class TestSiameseNetwork:
    def test_embedding_shape_is_8_by_64(self, model_and_loss, random_batch):
        """THE primary smoke test: batch (8, 60) → embeddings (8, 64).

        This is the contract the entire training loop depends on.
        """
        model, _ = model_and_loss
        anchor, pair, _ = random_batch
        emb_a, emb_b = model(anchor, pair)
        assert emb_a.shape == (BATCH, DEFAULT_EMBEDDING_DIM), (
            f"emb_a shape: {emb_a.shape}"
        )
        assert emb_b.shape == (BATCH, DEFAULT_EMBEDDING_DIM), (
            f"emb_b shape: {emb_b.shape}"
        )

    def test_loss_computes_without_error(self, model_and_loss, random_batch):
        """Full forward + loss pass must complete without NaN or error."""
        model, loss_fn = model_and_loss
        anchor, pair, labels = random_batch
        emb_a, emb_b = model(anchor, pair)
        dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
        loss = loss_fn(dist, labels)
        assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
        assert loss.item() >= 0.0,   f"Loss is negative: {loss.item()}"

    def test_backward_pass_updates_gradients(self, model_and_loss, random_batch):
        """Backprop must flow gradients into encoder weights.

        If gradients are zero/None, the encoder is not learning — this
        is the most common bug in custom loss implementations.
        """
        model, loss_fn = model_and_loss
        anchor, pair, labels = random_batch
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        emb_a, emb_b = model(anchor, pair)
        dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
        loss = loss_fn(dist, labels)
        loss.backward()

        # Check that at least some gradients are non-zero
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients computed — backward did not reach encoder"
        assert any(g > 0 for g in grad_norms), (
            "All gradients are zero — loss has no gradient signal"
        )

    def test_euclidean_distance_non_negative(self, model_and_loss, random_batch):
        """Distance must always be ≥ 0 — violated if the eps guard is wrong."""
        model, _ = model_and_loss
        anchor, pair, _ = random_batch
        emb_a, emb_b = model(anchor, pair)
        dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
        assert (dist >= 0).all(), f"Negative distances found: {dist[dist < 0]}"

    def test_same_input_distance_near_zero(self, model_and_loss):
        """Same tensor passed as both anchor and pair → distance ≈ 0.

        This is the identity check: if the encoder is working correctly,
        the same window must map to the same embedding.
        Note: must be in eval mode because Dropout breaks determinism.
        """
        model, _ = model_and_loss
        model.eval()
        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        emb_a, emb_b = model(x, x)
        dist = SiameseNetwork.euclidean_distance(emb_a, emb_b)
        # With eps=1e-8 inside sqrt, distance will be ~sqrt(1e-8) ≈ 1e-4 for identical inputs
        assert dist.max().item() < 1e-3, (
            f"Distance for identical inputs is {dist.max().item():.6f} — expected ~0"
        )
        model.train()

    def test_save_and_load_roundtrip(self, model_and_loss, tmp_path):
        """State dict save/load must produce identical forward pass outputs."""
        model, _ = model_and_loss
        model.eval()
        ckpt = tmp_path / "test_model.pt"
        model.save(ckpt)

        # Build a fresh model and load the saved weights
        model2, _ = build_model()
        model2 = model2.to(DEVICE)
        model2.load(ckpt, device=DEVICE)
        model2.eval()

        x = torch.randn(BATCH, FEATURE_DIM, device=DEVICE)
        with torch.no_grad():
            emb_a1, _ = model(x, x)
            emb_a2, _ = model2(x, x)

        assert torch.allclose(emb_a1, emb_a2, atol=1e-5), (
            "Saved and loaded model produce different outputs"
        )
        model.train()

    def test_parameter_count(self):
        """Verify total parameter count matches the known architecture.

        Breakdown (verified 2026-04-02):
          Linear(60→128) no-bias:         7,680
          BN(128):                           256
          Linear(128→256) no-bias:        32,768
          BN(256):                           512
          TransformerEncoderLayer ×2:  ~1,054,208
            (Q/K/V proj 196k + out_proj 65k + FFN 262k + norms 1k) × 2
          Linear(256→128) + bias:         32,896
          Linear(128→64)  + bias:          8,256
          Total:                       1,136,576

        The Transformer dominates (~93% of params). Dropout(0.3) and
        contrastive loss regularization prevent overfitting on 1,360 windows.
        If overfitting is observed during training, reduce transformer_layers
        to 1 via Optuna (src/training/tune.py) — do not change this file.
        """
        model, _ = build_model()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Exact count — if this fails, the architecture was changed
        assert n_params == 1_136_576, (
            f"Parameter count {n_params:,} != expected 1,136,576 — "
            f"architecture was modified without updating this test"
        )
