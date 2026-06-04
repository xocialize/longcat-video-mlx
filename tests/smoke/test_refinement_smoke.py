"""Smoke tests for refinement pipeline (no weights, no PT).

Verifies the trickier math pieces:
1. Trilinear (separable) resize gets shapes right + preserves a constant input.
2. _compute_padding rounds latents to multiples of bsa_latent_granularity (=4).
3. _pad_replicate_temporal extends front/back by replicated end frames.
4. _truncate_timesteps clips and inserts the threshold.
5. The full pipeline runs end-to-end with stubs.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest


def _import_smoke():
    from longcat_video.refinement import (
        RefinementPipelineConfig,
        LongCatVideoRefinementPipeline,
        _bilinear_resize_5d,
        _compute_padding,
        _pad_replicate_temporal,
    )
    return (
        RefinementPipelineConfig, LongCatVideoRefinementPipeline,
        _bilinear_resize_5d, _compute_padding, _pad_replicate_temporal,
    )


def test_imports():
    (Cfg, Pipeline, resize, compute_padding, pad_replicate) = _import_smoke()
    assert Cfg is not None
    assert Pipeline is not None


def test_config_defaults():
    Cfg, *_ = _import_smoke()
    cfg = Cfg()
    assert cfg.target_height == 720
    assert cfg.target_width == 1280
    assert cfg.t_thresh == 0.5
    assert cfg.spatial_refine_only is False
    assert cfg.bsa_latent_granularity == 4
    assert cfg.scheduler_shift == 12.0


def test_resize_5d_shape_only():
    _, _, resize, *_ = _import_smoke()
    x = mx.zeros((1, 3, 4, 60, 104))
    y = resize(x, new_T=8, new_H=120, new_W=208)
    assert y.shape == (1, 3, 8, 120, 208)


def test_resize_5d_constant_input_preserved():
    """A constant input should remain constant through trilinear resize."""
    _, _, resize, *_ = _import_smoke()
    x = mx.ones((1, 3, 4, 8, 8)) * 0.7
    y = resize(x, new_T=8, new_H=16, new_W=16)
    diff = float(mx.max(mx.abs(y - 0.7)))
    assert diff < 1e-5, f"Constant resize drifted by {diff}"


def test_compute_padding_no_cond():
    *_, compute_padding, _ = _import_smoke()
    # 23 noise frames, vae_temporal=4, bsa_gran=4
    # → num_noise_latents = ceil(23/4) = 6 → round up to multiple of 4 = 8
    # → num_noise_frames_added = 8*4 - 23 = 9
    (nc_lat, nc_added, nn_lat, nn_added) = compute_padding(
        num_cond_frames=0, num_noise_frames=23,
        vae_scale_temporal=4, bsa_latent_granularity=4,
    )
    assert nc_lat == 0
    assert nc_added == 0
    assert nn_lat == 8
    assert nn_added == 9


def test_compute_padding_with_cond():
    *_, compute_padding, _ = _import_smoke()
    # num_cond_frames=8, vae_temporal=4, bsa_gran=4
    # → num_cond_latents = 1 + ceil(7/4) = 1+2 = 3 → round up to 4
    # → num_cond_frames_added = 1 + (4-1)*4 - 8 = 13 - 8 = 5
    (nc_lat, nc_added, *_) = compute_padding(
        num_cond_frames=8, num_noise_frames=10,
        vae_scale_temporal=4, bsa_latent_granularity=4,
    )
    assert nc_lat == 4
    assert nc_added == 5


def test_pad_replicate_temporal():
    *_, pad_replicate = _import_smoke()
    # [B=1, C=3, T=4, H=2, W=2] — make the 4 frames distinguishable
    x = mx.stack([
        mx.full((1, 3, 2, 2), 10.0),
        mx.full((1, 3, 2, 2), 20.0),
        mx.full((1, 3, 2, 2), 30.0),
        mx.full((1, 3, 2, 2), 40.0),
    ], axis=2).reshape(1, 3, 4, 2, 2)
    y = pad_replicate(x, pad_front=2, pad_back=3)
    assert y.shape == (1, 3, 9, 2, 2)
    # First 2 frames should be 10, last 3 should be 40
    arr = np.asarray(y)
    assert (arr[:, :, 0] == 10.0).all()
    assert (arr[:, :, 1] == 10.0).all()
    assert (arr[:, :, 2] == 10.0).all()  # original first
    assert (arr[:, :, 5] == 40.0).all()  # original last
    assert (arr[:, :, 6] == 40.0).all()
    assert (arr[:, :, 8] == 40.0).all()


def test_truncate_timesteps():
    """Schedule should be clipped + threshold inserted at the head."""
    Cfg, Pipeline, *_ = _import_smoke()

    class StubScheduler:
        def __init__(self):
            self.timesteps = mx.array([900., 700., 500., 300., 100.])
        def set_timesteps(self, n): pass
        def step(self, n, t, lat): return lat

    class StubVAE:
        def encode(self, x): return mx.zeros((1, 16, 1, 1, 1))
        def normalize_latents(self, x): return x
        def denormalize_latents(self, x): return x
        def decode(self, x): return x

    class StubDiT:
        def __call__(self, lat, t, *a, **kw): return mx.zeros_like(lat)

    pipe = Pipeline(
        vae=StubVAE(), text_encoder=None, dit=StubDiT(),
        config=Cfg(), scheduler=StubScheduler(),
    )
    truncated = pipe._truncate_timesteps(pipe.scheduler.timesteps)
    arr = np.asarray(truncated).tolist()
    # t_thresh=0.5 → 500.0 inserted; then timesteps < 500 = [300, 100]
    assert arr == [500.0, 300.0, 100.0], f"Got {arr}"


def test_end_to_end_with_stubs():
    """Tiny end-to-end run through the pipeline with stub modules."""
    Cfg, Pipeline, *_ = _import_smoke()
    cfg = Cfg()
    cfg.target_height, cfg.target_width = 16, 32   # tiny
    cfg.num_sampling_steps = 4
    cfg.spatial_refine_only = True                 # keep frame count

    # Fake coarse video: 5 frames of 8x16 RGB
    stage1 = (np.zeros((5, 8, 16, 3)) * 200).astype(np.uint8)

    class StubVAE:
        def encode(self, x):
            # x: [1, 3, T_pad, 16, 32] — encode to [1, 16, T_lat, 2, 4]
            T = int(x.shape[2])
            T_lat = max(1, T // 4)
            return mx.zeros((1, 16, T_lat, 2, 4))
        def normalize_latents(self, x): return x
        def denormalize_latents(self, x): return x
        def decode(self, x):
            T_lat = int(x.shape[2])
            return mx.zeros((1, 3, T_lat * 4, 16, 32))

    class StubDiT:
        def __call__(self, lat, t, *a, **kw): return mx.zeros_like(lat)

    class StubScheduler:
        def __init__(self):
            self.timesteps = mx.array([900., 700., 500., 300., 100.])
        def set_timesteps(self, n): pass
        def step(self, n, t, lat): return lat

    pipe = Pipeline(
        vae=StubVAE(), text_encoder=None, dit=StubDiT(),
        config=cfg, scheduler=StubScheduler(),
    )

    text = mx.zeros((1, 1, 16, 4096))
    mask = mx.ones((1, 16))
    video = pipe(stage1_video_np=stage1, text_embeds=text, text_mask=mask,
                 num_cond_frames=0, seed=0)
    assert video.shape[0] == 1
    assert video.shape[1] == 3
    # spatial_refine_only=True → T_out has the upsample size = 5 (same frame count),
    # then pad to multiple-of-4 latent (5 noise frames → ceil(5/4)=2 → round to 4 latents;
    # 4*4 = 16 padded raw frames). Front padding sliced off → end up with at least 5 frames.
    assert video.shape[2] >= 5
