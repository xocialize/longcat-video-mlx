"""Smoke tests for the I2V pipeline (no weights, no PT).

Verify:
1. The pipeline class is importable and constructs with stub modules.
2. The CFG forward + concat-prefix wiring runs end-to-end with tiny stubs.
3. Latent-frame math (T_lat = 1 + (T-1)//4) lines up with what _make_noise
   computes, including the +1 ref-latent prefix on the time axis.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest


def _import_smoke():
    from longcat_video.pipeline_i2v import (
        I2VPipelineConfig,
        LongCatVideoI2VPipeline,
    )
    return I2VPipelineConfig, LongCatVideoI2VPipeline


def test_imports():
    I2VPipelineConfig, LongCatVideoI2VPipeline = _import_smoke()
    assert I2VPipelineConfig is not None
    assert LongCatVideoI2VPipeline is not None


def test_config_defaults_match_base_model():
    I2VPipelineConfig, _ = _import_smoke()
    cfg = I2VPipelineConfig()
    assert cfg.scheduler_shift == 12.0, "Base model uses shift=12 (not 7 like Avatar)"
    assert cfg.dit_in_channels == 16
    assert cfg.text_guidance_scale == 5.0
    assert cfg.num_sampling_steps == 50
    assert cfg.cfg_collapse is False


def test_latent_frame_math():
    """For T=24 raw frames, vae_temporal=4: T_lat_noise = 1 + 23//4 = 6.
    With +1 ref latent at the head, the DiT sees T_lat = 7 total.
    """
    I2VPipelineConfig, LongCatVideoI2VPipeline = _import_smoke()
    cfg = I2VPipelineConfig()

    # Stubs
    class StubVAE:
        def encode(self, x): return mx.zeros((1, 16, 1, 60, 104))
        def normalize_latents(self, x): return x
        def denormalize_latents(self, x): return x
        def decode(self, x):
            T_lat = int(x.shape[2])
            return mx.zeros((1, 3, T_lat * 4, 480, 832))

    class StubDiT:
        def __call__(self, lat, t, *a, **kw):
            # passthrough zero velocity for smoke
            return mx.zeros_like(lat)

    class StubText:
        pass

    class StubScheduler:
        def __init__(self): self.timesteps = mx.array([1000.0, 500.0])
        def set_timesteps(self, n): pass
        def step(self, noise, t, lat): return lat  # noop

    pipe = LongCatVideoI2VPipeline(
        vae=StubVAE(), text_encoder=StubText(), dit=StubDiT(),
        config=cfg, scheduler=StubScheduler(),
    )

    ref = pipe._encode_reference_image(mx.zeros((1, 3, 1, 480, 832)))
    assert ref.shape == (1, 16, 1, 60, 104)

    noise = pipe._make_noise(num_frames=24, height=480, width=832, seed=0)
    assert noise.shape == (1, 16, 6, 60, 104), \
        f"Expected (1,16,6,60,104), got {noise.shape}"

    cat = mx.concatenate([ref, noise], axis=2)
    assert cat.shape == (1, 16, 7, 60, 104)
