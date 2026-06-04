"""Smoke tests for the Continuation pipeline (no weights, no PT).

Verify:
1. Imports cleanly.
2. Config defaults match the base model (shift=12, 50 steps).
3. Prefix encoding + noise concat math:
   - For T_cond=8 raw cond frames, vae_temporal=4: T_cond_lat = 1 + 7//4 = 2
   - For num_new_frames=24, vae_temporal=4: T_new_lat = 1 + 23//4 = 6
   - Concat at time axis = T_lat 8 — what the DiT sees.
"""

from __future__ import annotations

import mlx.core as mx
import pytest


def _import_smoke():
    from longcat_video.pipeline_continuation import (
        ContinuationPipelineConfig,
        LongCatVideoContinuationPipeline,
    )
    return ContinuationPipelineConfig, LongCatVideoContinuationPipeline


def test_imports():
    Config, Pipeline = _import_smoke()
    assert Config is not None
    assert Pipeline is not None


def test_config_defaults_match_base_model():
    Config, _ = _import_smoke()
    cfg = Config()
    assert cfg.scheduler_shift == 12.0
    assert cfg.dit_in_channels == 16
    assert cfg.text_guidance_scale == 5.0
    assert cfg.num_sampling_steps == 50
    assert cfg.cfg_collapse is False
    # Continuation-specific: num_new_frames (not num_frames)
    assert cfg.num_new_frames == 24


def test_prefix_concat_math():
    """T_cond=8 raw → 2 cond latent frames (1 + 7//4).
       num_new=24 raw → 6 new latent frames (1 + 23//4).
       Concat → 8 latent frames total at time axis.
    """
    Config, Pipeline = _import_smoke()
    cfg = Config()

    class StubVAE:
        def encode(self, x):
            # Cond clip: T_cond=8 raw → T_cond_lat = 1 + 7//4 = 2
            T = int(x.shape[2])
            T_lat = 1 + (T - 1) // 4
            return mx.zeros((1, 16, T_lat, x.shape[3] // 8, x.shape[4] // 8))
        def normalize_latents(self, x): return x
        def denormalize_latents(self, x): return x
        def decode(self, x):
            T_lat = int(x.shape[2])
            return mx.zeros((1, 3, T_lat * 4, 480, 832))

    class StubDiT:
        def __call__(self, lat, t, *a, **kw): return mx.zeros_like(lat)

    class StubScheduler:
        def __init__(self): self.timesteps = mx.array([1000.0])
        def set_timesteps(self, n): pass
        def step(self, noise, t, lat): return lat

    pipe = Pipeline(
        vae=StubVAE(), text_encoder=None, dit=StubDiT(),
        config=cfg, scheduler=StubScheduler(),
    )

    prefix = mx.zeros((1, 3, 8, 480, 832))
    cond_lat = pipe._encode_prefix_clip(prefix)
    assert cond_lat.shape == (1, 16, 2, 60, 104), \
        f"Expected (1,16,2,60,104), got {cond_lat.shape}"

    noise = pipe._make_noise(num_new_frames=24, height=480, width=832, seed=0)
    assert noise.shape == (1, 16, 6, 60, 104)

    cat = mx.concatenate([cond_lat, noise], axis=2)
    assert cat.shape == (1, 16, 8, 60, 104)


def test_return_full_video_flag():
    """return_full_video=False slices the cond prefix out before decode."""
    Config, Pipeline = _import_smoke()
    cfg = Config()
    cfg.num_sampling_steps = 1

    class StubVAE:
        def encode(self, x):
            T = int(x.shape[2])
            T_lat = 1 + (T - 1) // 4
            return mx.zeros((1, 16, T_lat, x.shape[3] // 8, x.shape[4] // 8))
        def normalize_latents(self, x): return x
        def denormalize_latents(self, x): return x
        def decode(self, x):
            T_lat = int(x.shape[2])
            return mx.zeros((1, 3, T_lat * 4, 480, 832))

    class StubDiT:
        def __call__(self, lat, t, *a, **kw): return mx.zeros_like(lat)

    class StubScheduler:
        def __init__(self): self.timesteps = [mx.array(1000.0)]
        def set_timesteps(self, n): pass
        def step(self, noise, t, lat): return lat

    pipe = Pipeline(
        vae=StubVAE(), text_encoder=None, dit=StubDiT(),
        config=cfg, scheduler=StubScheduler(),
    )

    prefix = mx.zeros((1, 3, 8, 480, 832))
    text = mx.zeros((1, 1, 16, 4096))
    mask = mx.ones((1, 16))

    full = pipe(
        prefix_video=prefix, text_embeds=text, text_mask=mask,
        uncond_embeds=text, uncond_mask=mask,
        num_new_frames=24, height=480, width=832, seed=0,
        return_full_video=True,
    )
    new_only = pipe(
        prefix_video=prefix, text_embeds=text, text_mask=mask,
        uncond_embeds=text, uncond_mask=mask,
        num_new_frames=24, height=480, width=832, seed=0,
        return_full_video=False,
    )
    # full = 8 lat → 32 decoded; new_only = 6 lat → 24 decoded
    assert full.shape[2] == 32
    assert new_only.shape[2] == 24
