"""Pipeline-level CFG combiner for the base LongCat-Video model.

Simpler than Avatar's 3-pass disentangled CFG: base uses standard
2-pass text CFG (cond + uncond). When `cfg_step_lora` is merged into the
DiT, CFG collapses into a single forward pass (the LoRA learns to predict
the CFG-combined output directly) — caller passes `text_guidance_scale=0`
to bypass the second uncond pass.

PT reference: `run_demo_text_to_video.py` (the `do_classifier_free_guidance`
branch in the LongCat-Video upstream repo).
"""

from __future__ import annotations

import mlx.core as mx


def cfg_combine(
    noise_pred_cond: mx.array,
    noise_pred_uncond: mx.array,
    text_guidance_scale: float = 5.0,
) -> mx.array:
    """Standard 2-pass classifier-free guidance.

    Formula:
        noise_pred = uncond + scale * (cond - uncond)

    Defaults to scale=5.0 — typical for video diffusion at base settings.
    When `cfg_step_lora` is merged, set scale=0.0 to skip the (cond-uncond)
    correction term (the LoRA has absorbed it).
    """
    return noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond)


def flip_velocity_for_scheduler(noise_pred: mx.array) -> mx.array:
    """LongCat DiT predicts negative velocity (`ε - x_0`). Flip the sign
    before handing to `FlowMatchEulerDiscreteScheduler.step()`.

    Same convention as the Avatar pipeline — easy to forget, ports always
    blow up here first if it's missed.
    """
    return -noise_pred


def cfg_split_outputs(noise_pred_2batch: mx.array) -> tuple[mx.array, mx.array]:
    """Split a doubled-batch CFG forward output into `(uncond, cond)`.

    The pipeline stacks `[uncond_text, pos_text]` so the first half is
    `noise_pred_uncond` and the second is `noise_pred_cond`. Same ordering
    as Avatar's `cfg_split_outputs`.
    """
    uncond, cond = mx.split(noise_pred_2batch, 2, axis=0)
    return uncond, cond
