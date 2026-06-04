"""Smoke tests for the Interactive Video orchestrator."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _import_smoke():
    from longcat_video.pipeline_interactive import (
        InteractivePipelineConfig,
        LongCatVideoInteractivePipeline,
    )
    return InteractivePipelineConfig, LongCatVideoInteractivePipeline


def test_imports():
    Cfg, Pipeline = _import_smoke()
    assert Cfg is not None
    assert Pipeline is not None


def test_empty_prompts_raises():
    Cfg, Pipeline = _import_smoke()
    pipe = Pipeline(vae=None, text_encoder=None, dit=None, config=Cfg())
    with pytest.raises(ValueError, match="at least 1 prompt"):
        pipe(prompts_encoded=[])


def test_per_segment_prompts_received():
    """Each segment should be called with its own prompt embeddings.
    Verify by stubbing t2v/cont to capture and inspect the arg dict.
    """
    Cfg, Pipeline = _import_smoke()
    cfg = Cfg()
    cfg.num_frames_per_segment = 8
    cfg.num_cond_frames = 4
    cfg.height = 16
    cfg.width = 32

    received_te_ids = []

    class StubT2V:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            received_te_ids.append(id(kwargs["text_embeds"]))
            return mx.zeros((1, 3, kwargs["num_frames"], cfg.height, cfg.width))

    class StubCont:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            received_te_ids.append(id(kwargs["text_embeds"]))
            return mx.zeros((1, 3, kwargs["num_new_frames"], cfg.height, cfg.width))

    pipe = Pipeline(vae=None, text_encoder=None, dit=None, config=cfg)
    pipe.t2v = StubT2V()
    pipe.continuation = StubCont()

    # 3 distinct prompts → 3 distinct text_embeds objects → 3 distinct id()s
    prompts = []
    for i in range(3):
        te = mx.full((1, 1, 16, 4096), float(i + 1))
        tm = mx.ones((1, 1, 1, 16))
        ue = mx.zeros((1, 1, 16, 4096))
        um = mx.ones((1, 1, 1, 16))
        prompts.append((te, tm, ue, um))

    result = pipe(prompts_encoded=prompts)
    assert result.shape == (8 + 4 + 4, cfg.height, cfg.width, 3)
    # Each call should have received a DIFFERENT text_embeds object
    assert len(set(received_te_ids)) == 3, \
        f"Expected 3 unique prompt embeds, got {len(set(received_te_ids))}"


def test_callback_with_labels():
    Cfg, Pipeline = _import_smoke()
    cfg = Cfg()
    cfg.num_frames_per_segment = 8
    cfg.num_cond_frames = 4
    cfg.height = 16
    cfg.width = 32

    class StubT2V:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            return mx.zeros((1, 3, kwargs["num_frames"], cfg.height, cfg.width))
    class StubCont:
        def __init__(self, *a, **kw): pass
        def __call__(self, **kwargs):
            return mx.zeros((1, 3, kwargs["num_new_frames"], cfg.height, cfg.width))

    pipe = Pipeline(vae=None, text_encoder=None, dit=None, config=cfg)
    pipe.t2v = StubT2V()
    pipe.continuation = StubCont()

    prompts = [(mx.zeros((1, 1, 16, 4096)), mx.ones((1, 1, 1, 16)),
                mx.zeros((1, 1, 16, 4096)), mx.ones((1, 1, 1, 16)))
               for _ in range(2)]
    labels = ["A cat appears", "The cat starts running"]
    fired = []
    pipe(prompts_encoded=prompts, prompt_labels=labels,
         on_segment_done=lambda i, label, frames: fired.append((i, label)))
    assert fired == [(0, "A cat appears"), (1, "The cat starts running")]
