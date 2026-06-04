"""Smoke tests for longcat_video/guidance.py."""

from __future__ import annotations

import mlx.core as mx

from longcat_video.guidance import (
    cfg_combine,
    cfg_split_outputs,
    flip_velocity_for_scheduler,
)


def test_cfg_combine_formula_matches_expected():
    """Standard CFG: uncond + scale * (cond - uncond). At scale=5.0 with
    cond=2, uncond=1, expected = 1 + 5*(2-1) = 6.
    """
    cond = mx.array([2.0, 2.0, 2.0])
    uncond = mx.array([1.0, 1.0, 1.0])
    out = cfg_combine(cond, uncond, text_guidance_scale=5.0)
    mx.eval(out)
    assert out.tolist() == [6.0, 6.0, 6.0]


def test_cfg_combine_at_scale_zero_returns_uncond():
    """scale=0 collapses to just uncond — cfg_step_lora-merged mode."""
    cond = mx.array([5.0])
    uncond = mx.array([1.0])
    out = cfg_combine(cond, uncond, text_guidance_scale=0.0)
    mx.eval(out)
    assert out.tolist() == [1.0]


def test_flip_velocity_negates():
    """LongCat DiT outputs -v; pipeline flips before scheduler.step."""
    v = mx.array([1.5, -2.0, 3.0])
    flipped = flip_velocity_for_scheduler(v)
    mx.eval(flipped)
    assert flipped.tolist() == [-1.5, 2.0, -3.0]


def test_cfg_split_outputs_unstacks_uncond_first():
    """Pipeline stacks [uncond, cond]. Split returns (uncond, cond)."""
    uncond = mx.array([1.0, 2.0, 3.0]).reshape(1, 3)
    cond = mx.array([4.0, 5.0, 6.0]).reshape(1, 3)
    stacked = mx.concatenate([uncond, cond], axis=0)
    u, c = cfg_split_outputs(stacked)
    mx.eval(u, c)
    assert u.flatten().tolist() == [1.0, 2.0, 3.0]
    assert c.flatten().tolist() == [4.0, 5.0, 6.0]
