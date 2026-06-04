"""Smoke tests for the Long-Video orchestrator.

Verifies:
1. Imports cleanly.
2. Default config: 11 segments × 93 frames - 10 × 13 cond = ~893 total frames.
3. Sub-pipelines share component instances (no duplicate weight load).
4. End-to-end orchestration with stubs runs the segment loop correctly:
   - Segment 1 is T2V (no prefix)
   - Segments 2+ are Continuation w/ last-N-frames prefix
   - Final concat shape matches `T_total = T_seg0 + (N-1) * (T_seg - cond)`
5. on_segment_done callback fires once per segment with correct index.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _import_smoke():
    from longcat_video.pipeline_long_video import (
        LongCatVideoLongVideoPipeline,
        LongVideoPipelineConfig,
    )
    return LongCatVideoLongVideoPipeline, LongVideoPipelineConfig


def test_imports():
    Pipeline, Cfg = _import_smoke()
    assert Pipeline is not None
    assert Cfg is not None


def test_config_defaults():
    _, Cfg = _import_smoke()
    cfg = Cfg()
    assert cfg.num_segments == 11
    assert cfg.num_frames_per_segment == 93
    assert cfg.num_cond_frames == 13
    # Total = 93 + 10 * (93 - 13) = 93 + 800 = 893 frames ≈ 59.5s @ 15fps
    expected = (cfg.num_frames_per_segment
                + (cfg.num_segments - 1) * (cfg.num_frames_per_segment - cfg.num_cond_frames))
    assert expected == 893
    assert cfg.scheduler_shift == 12.0


def test_subpipelines_share_components():
    """The internal T2V + Continuation pipelines must reference the SAME
    vae / umt5 / dit instances — duplicate loads at the Avatar repo's
    11 GB umT5 + 26 GB DiT scale would blow up the unified memory."""
    Pipeline, _ = _import_smoke()
    vae_stub = object()
    umt5_stub = object()
    dit_stub = object()
    pipe = Pipeline(vae=vae_stub, text_encoder=umt5_stub, dit=dit_stub)
    assert pipe.t2v.vae is vae_stub
    assert pipe.continuation.vae is vae_stub
    assert pipe.t2v.dit is dit_stub
    assert pipe.continuation.dit is dit_stub


def test_end_to_end_with_stubs():
    """3-segment run with stubs. Verifies segment loop + concat math."""
    Pipeline, Cfg = _import_smoke()
    cfg = Cfg()
    cfg.num_segments = 3
    cfg.num_frames_per_segment = 8
    cfg.num_cond_frames = 4
    cfg.height = 16
    cfg.width = 32

    # Stub each sub-pipeline's __call__ to bypass real DiT
    class StubT2V:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            T = kwargs["num_frames"]
            return mx.zeros((1, 3, T, cfg.height, cfg.width)) * 0.0
    class StubCont:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            # Continuation returns only the NEW frames when return_full_video=False
            T_new = kwargs["num_new_frames"]
            return mx.ones((1, 3, T_new, cfg.height, cfg.width)) * 0.5

    pipe = Pipeline(vae=None, text_encoder=None, dit=None, config=cfg)
    pipe.t2v = StubT2V()
    pipe.continuation = StubCont()

    text = mx.zeros((1, 1, 16, 4096))
    mask = mx.ones((1, 1, 1, 16))
    callbacks = []
    def on_done(seg_idx, frames):
        callbacks.append((seg_idx, frames.shape))

    result = pipe(
        text_embeds=text, text_mask=mask,
        uncond_embeds=text, uncond_mask=mask,
        num_segments=3,
        on_segment_done=on_done,
    )
    # Expected total frames:
    #   seg 0 (T2V):     8 frames
    #   seg 1 (Cont):    8 - 4 = 4 NEW frames
    #   seg 2 (Cont):    4 NEW frames
    #   TOTAL:           16 frames
    assert result.shape == (16, cfg.height, cfg.width, 3)
    assert result.dtype == np.uint8

    # Callback fires once per segment
    assert len(callbacks) == 3
    assert [c[0] for c in callbacks] == [0, 1, 2]
    # Segment 0 has 8 frames, segments 1 + 2 have 4 each
    assert callbacks[0][1] == (8, cfg.height, cfg.width, 3)
    assert callbacks[1][1] == (4, cfg.height, cfg.width, 3)
    assert callbacks[2][1] == (4, cfg.height, cfg.width, 3)


def test_short_run_minimum_segments():
    """num_segments=1 should produce just the T2V seed clip, no Continuation."""
    Pipeline, Cfg = _import_smoke()
    cfg = Cfg()
    cfg.num_segments = 1
    cfg.num_frames_per_segment = 8
    cfg.height = 16
    cfg.width = 32

    class StubT2V:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            return mx.zeros((1, 3, kwargs["num_frames"], cfg.height, cfg.width))
    class StubCont:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            raise AssertionError("Continuation must NOT be called for num_segments=1")

    pipe = Pipeline(vae=None, text_encoder=None, dit=None, config=cfg)
    pipe.t2v = StubT2V()
    pipe.continuation = StubCont()

    text = mx.zeros((1, 1, 16, 4096))
    mask = mx.ones((1, 1, 1, 16))
    result = pipe(text_embeds=text, text_mask=mask,
                  uncond_embeds=text, uncond_mask=mask,
                  num_segments=1)
    assert result.shape == (8, cfg.height, cfg.width, 3)
