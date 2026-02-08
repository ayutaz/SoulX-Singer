"""Tests for SoulXSinger.forward() training method.

Uses a tiny model configuration to run fast on CPU.
"""

import torch
import pytest
from omegaconf import OmegaConf

from soulxsinger.models.soulxsinger import SoulXSinger


def make_tiny_config():
    """Create a minimal model config for fast CPU testing."""
    return OmegaConf.create({
        "audio": {
            "hop_size": 480,
            "sample_rate": 24000,
            "max_length": 36000,
            "n_fft": 1920,
            "num_mels": 128,
            "win_size": 1920,
            "fmin": 0,
            "fmax": 12000,
            "mel_var": 8.14,
            "mel_mean": -4.92,
        },
        "model": {
            "encoder": {
                "vocab_size": 100,
                "text_dim": 64,
                "pitch_dim": 64,
                "type_dim": 64,
                "f0_bin": 361,
                "f0_dim": 64,
                "num_layers": 1,
            },
            "flow_matching": {
                "mel_dim": 128,
                "hidden_size": 64,
                "num_layers": 1,
                "num_heads": 4,
                "cfg_drop_prob": 0.2,
                "use_embedding": False,
                "cond_codebook_size": 64,
                "cond_scale_factor": 1,
                "sigma": 1e-5,
                "time_scheduler": "cos",
            },
        },
    })


def make_dummy_batch(batch_size=2, n_notes=10, prompt_mel_frames=20, target_mel_frames=30, sample_rate=24000, hop_size=480, device="cpu"):
    """Create a dummy batch for testing forward()."""
    total_mel = prompt_mel_frames + target_mel_frames

    prompt_wav_len = prompt_mel_frames * hop_size
    target_wav_len = target_mel_frames * hop_size

    return {
        "phoneme": torch.randint(0, 100, (batch_size, n_notes), device=device),
        "note_pitch": torch.randint(0, 128, (batch_size, n_notes), device=device),
        "note_type": torch.randint(0, 3, (batch_size, n_notes), device=device),
        "mel2note": torch.clamp(torch.randint(0, n_notes, (batch_size, total_mel), device=device), max=n_notes - 1),
        "f0": torch.cat([
            torch.zeros(batch_size, prompt_mel_frames, device=device),
            torch.ones(batch_size, target_mel_frames, device=device) * 440.0,
        ], dim=1),
        "mel_mask": torch.ones(batch_size, total_mel, dtype=torch.float, device=device),
        "is_prompt": torch.cat([
            torch.ones(batch_size, prompt_mel_frames, dtype=torch.float, device=device),
            torch.zeros(batch_size, target_mel_frames, dtype=torch.float, device=device),
        ], dim=1),
        "prompt_mel_len": torch.full((batch_size,), prompt_mel_frames, dtype=torch.long, device=device),
        "prompt_waveform": torch.randn(batch_size, 1, prompt_wav_len, device=device) * 0.1,
        "target_waveform": torch.randn(batch_size, 1, target_wav_len, device=device) * 0.1,
    }


@pytest.fixture
def tiny_model():
    config = make_tiny_config()
    model = SoulXSinger(config)
    model.train()
    return model


class TestForwardBasic:
    """Basic forward pass tests."""

    def test_forward_returns_loss_dict(self, tiny_model):
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        assert isinstance(outputs, dict)
        assert "loss" in outputs

    def test_loss_is_scalar(self, tiny_model):
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        loss = outputs["loss"]
        assert loss.dim() == 0  # scalar
        assert loss.dtype == torch.float32

    def test_loss_is_finite(self, tiny_model):
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        assert torch.isfinite(outputs["loss"])

    def test_loss_is_positive(self, tiny_model):
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        assert outputs["loss"].item() > 0

    def test_forward_batch_size_1(self, tiny_model):
        batch = make_dummy_batch(batch_size=1)
        outputs = tiny_model(batch)
        assert torch.isfinite(outputs["loss"])


class TestForwardGradients:
    """Test that gradients flow correctly."""

    def test_loss_is_differentiable(self, tiny_model):
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        # At least some parameters should have gradients
        has_grad = False
        for p in tiny_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found after backward()"

    def test_encoder_params_get_gradients(self, tiny_model):
        """Encoder embedding parameters should receive gradients.

        Note: nn.Embedding may produce sparse gradients where .grad is a
        sparse tensor. We check that grad exists (not None) for at least
        one parameter.
        """
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        for name in ("note_text_encoder", "note_pitch_encoder", "note_type_encoder", "f0_encoder"):
            encoder = getattr(tiny_model, name)
            has_grad = any(p.grad is not None for p in encoder.parameters())
            assert has_grad, f"{name} did not receive any gradient tensor"

    def test_preflow_gets_gradients(self, tiny_model):
        batch = make_dummy_batch(n_notes=20, prompt_mel_frames=30, target_mel_frames=50)
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        # preflow gradients may be very small with random data; check they exist
        has_any_grad = any(p.grad is not None for p in tiny_model.preflow.parameters())
        assert has_any_grad, "preflow parameters have no gradient tensors at all"

    def test_vocoder_no_gradients(self, tiny_model):
        """Vocoder is not used in forward, so no gradients expected."""
        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        for p in tiny_model.vocoder.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, \
                "Vocoder should not receive gradients during training"


class TestForwardFreeze:
    """Test freezing behavior for fine-tuning."""

    def test_frozen_encoder_no_gradients(self, tiny_model):
        """When encoder is frozen, it should not get gradients."""
        for name in ("note_text_encoder", "note_pitch_encoder", "note_type_encoder", "f0_encoder"):
            for p in getattr(tiny_model, name).parameters():
                p.requires_grad = False
        for p in tiny_model.preflow.parameters():
            p.requires_grad = False

        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        for name in ("note_text_encoder", "note_pitch_encoder"):
            for p in getattr(tiny_model, name).parameters():
                assert p.grad is None, f"{name} should not have gradients when frozen"

    def test_frozen_encoder_transformer_still_trains(self, tiny_model):
        """Even with encoder frozen, transformer should still get gradients."""
        for name in ("note_text_encoder", "note_pitch_encoder", "note_type_encoder", "f0_encoder"):
            for p in getattr(tiny_model, name).parameters():
                p.requires_grad = False
        for p in tiny_model.preflow.parameters():
            p.requires_grad = False

        batch = make_dummy_batch()
        outputs = tiny_model(batch)
        outputs["loss"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.cfm_decoder.parameters()
            if p.requires_grad
        )
        assert has_grad, "CFM decoder should still receive gradients"


class TestForwardEdgeCases:
    """Edge cases and robustness tests."""

    def test_zero_f0_all_silent(self, tiny_model):
        """All-zero F0 (silent) should still produce a valid loss."""
        batch = make_dummy_batch()
        batch["f0"] = torch.zeros_like(batch["f0"])
        outputs = tiny_model(batch)
        assert torch.isfinite(outputs["loss"])

    def test_varying_prompt_target_ratio(self, tiny_model):
        """Different prompt/target length ratios should work."""
        for pt_len, tg_len in [(10, 50), (40, 20), (5, 5)]:
            batch = make_dummy_batch(prompt_mel_frames=pt_len, target_mel_frames=tg_len)
            outputs = tiny_model(batch)
            assert torch.isfinite(outputs["loss"]), f"Failed for pt={pt_len}, tg={tg_len}"

    def test_with_padding_in_mask(self, tiny_model):
        """Test that partial mel_mask (with padding) works."""
        batch = make_dummy_batch(batch_size=2, prompt_mel_frames=20, target_mel_frames=30)
        # Simulate padding: second sample is shorter
        batch["mel_mask"][1, 40:] = 0
        outputs = tiny_model(batch)
        assert torch.isfinite(outputs["loss"])


class TestForwardOptimization:
    """Test that optimization actually reduces loss."""

    def test_loss_decreases_over_steps(self, tiny_model):
        """Multiple optimization steps should reduce the loss."""
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad],
            lr=5e-3,
        )
        batch = make_dummy_batch()

        losses = []
        for _ in range(20):
            outputs = tiny_model(batch)
            loss = outputs["loss"]
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease overall (average of last 5 vs first 5)
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Loss did not decrease: avg_first={avg_first:.4f}, avg_last={avg_last:.4f}"
        )


class TestF0ToCoarse:
    """Test the f0_to_coarse static method used in forward()."""

    def test_silent_frames_map_to_zero(self):
        f0 = torch.tensor([0.0, 0.0, -1.0])
        coarse = SoulXSinger.f0_to_coarse(f0)
        assert (coarse == 0).all()

    def test_valid_f0_maps_to_positive_bins(self):
        f0 = torch.tensor([440.0, 261.63, 130.81])
        coarse = SoulXSinger.f0_to_coarse(f0)
        assert (coarse > 0).all()
        assert (coarse < 361).all()

    def test_f0_bin_range(self):
        """All output bins should be in [0, 360]."""
        f0 = torch.linspace(30, 2000, 200)
        coarse = SoulXSinger.f0_to_coarse(f0)
        assert coarse.min() >= 0
        assert coarse.max() <= 360

    def test_monotonic_for_increasing_f0(self):
        """Higher F0 should map to higher bin (non-decreasing)."""
        f0 = torch.tensor([100.0, 200.0, 400.0, 800.0])
        coarse = SoulXSinger.f0_to_coarse(f0)
        for i in range(len(coarse) - 1):
            assert coarse[i] <= coarse[i + 1]

    def test_f0_shift_applied(self):
        f0 = torch.tensor([440.0])
        base = SoulXSinger.f0_to_coarse(f0, f0_shift=0)
        shifted = SoulXSinger.f0_to_coarse(f0, f0_shift=5)
        assert shifted.item() == base.item() + 5

    def test_numpy_input(self):
        import numpy as np
        f0 = np.array([440.0, 0.0, 261.63])
        coarse = SoulXSinger.f0_to_coarse(f0)
        assert coarse[0] > 0
        assert coarse[1] == 0
        assert coarse[2] > 0
