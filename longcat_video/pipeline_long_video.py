"""Long-Video orchestration: chain T2V + N × Continuation segments.

Mirrors upstream's `run_demo_long_video.py:104-130` chaining pattern:

1. Segment 1: T2V seed clip (e.g. 93 frames @ 15fps ≈ 6.2s)
2. Segments 2..N: Continuation, conditioned on the last
   `num_cond_frames` of the previous segment. The output strips that
   prefix so each segment contributes `num_frames - num_cond_frames`
   NEW frames to the running video.

For an 11-segment run with `num_frames=93, num_cond_frames=13`:
total = 93 + 10 * (93 - 13) = 893 frames ≈ 59.5s at 15 fps.

This is **pure-Python orchestration** on top of the existing T2V and
Continuation pipelines. No DiT changes, no new attention paths. Memory
stays bounded because each segment's decoded video is kept only long
enough to feed the next segment as conditioning, then dropped.

For per-segment prompts (Interactive Video), see B5.2 — same chaining
shape, different prompt per iteration.

For refinement, take the long-video output and feed it to
`scripts/run_refine.py` (B3.1) — refinement runs once over the full
concatenation, not per-segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.models.umt5 import UMT5EncoderModel
from longcat_video.pipeline_continuation import (
    ContinuationPipelineConfig,
    LongCatVideoContinuationPipeline,
)
from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline, T2VPipelineConfig


@dataclass
class LongVideoPipelineConfig:
    """Config for Long-Video orchestration."""

    # Per-segment defaults
    num_segments: int = 11               # ~1 minute @ 15fps if num_frames=93
    num_frames_per_segment: int = 93     # each Continuation step generates this many
    num_cond_frames: int = 13            # last-N frames carried as cond into next segment

    # Resolution
    height: int = 480
    width: int = 832
    target_fps: int = 15

    # Per-pipeline settings (inherit defaults but overridable)
    num_sampling_steps: int = 50
    text_guidance_scale: float = 5.0
    scheduler_shift: float = 12.0
    cfg_collapse: bool = False           # True when cfg_step_lora merged

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


def _np_uint8_to_mx_video(arr: np.ndarray) -> mx.array:
    """[T, H, W, 3] uint8 → [1, 3, T, H, W] mx in [-1, 1]."""
    a = arr.astype(np.float32) / 127.5 - 1.0
    a = a.transpose(3, 0, 1, 2)[None, :]
    return mx.array(a)


def _mx_video_to_np_uint8(video: mx.array) -> np.ndarray:
    """[1, 3, T, H, W] mx in [-1, 1] → [T, H, W, 3] uint8."""
    return (
        np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5
    ).clip(0, 255).astype(np.uint8)


class LongCatVideoLongVideoPipeline:
    """Long-Video pipeline = orchestrated T2V + chained Continuation.

    Holds references to the three core components (VAE / umT5 / DiT) plus
    a T2V and Continuation pipeline that share them. The two sub-pipelines
    share the same component instances — there's no duplicate weight load.

    Usage:
        pipeline = LongCatVideoLongVideoPipeline(vae, umt5, dit)
        all_frames = pipeline(
            text_embeds, text_mask, uncond_embeds, uncond_mask,
            num_segments=11,
        )
        # all_frames: np.ndarray [T_total, H, W, 3] uint8
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[LongVideoPipelineConfig] = None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or LongVideoPipelineConfig()

        # Build sub-pipelines with consistent settings
        t2v_cfg = T2VPipelineConfig(
            num_frames=self.config.num_frames_per_segment,
            target_fps=self.config.target_fps,
            num_sampling_steps=self.config.num_sampling_steps,
            text_guidance_scale=self.config.text_guidance_scale,
            scheduler_shift=self.config.scheduler_shift,
            cfg_collapse=self.config.cfg_collapse,
            vae_scale_temporal=self.config.vae_scale_temporal,
            vae_scale_spatial=self.config.vae_scale_spatial,
        )
        cont_cfg = ContinuationPipelineConfig(
            num_new_frames=self.config.num_frames_per_segment - self.config.num_cond_frames,
            target_fps=self.config.target_fps,
            num_sampling_steps=self.config.num_sampling_steps,
            text_guidance_scale=self.config.text_guidance_scale,
            scheduler_shift=self.config.scheduler_shift,
            cfg_collapse=self.config.cfg_collapse,
            vae_scale_temporal=self.config.vae_scale_temporal,
            vae_scale_spatial=self.config.vae_scale_spatial,
        )
        self.t2v = LongCatVideoT2VPipeline(vae, text_encoder, dit, config=t2v_cfg)
        self.continuation = LongCatVideoContinuationPipeline(
            vae, text_encoder, dit, config=cont_cfg,
        )

    def __call__(
        self,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_segments: Optional[int] = None,
        seed: int = 0,
        on_segment_done: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Run the long-video chain.

        Args:
            text_embeds:     [1, 1, N_text, 4096] from umT5 (positive)
            text_mask:       [1, 1, 1, N_text] DiT-shaped mask
            uncond_embeds:   [1, 1, N_text, 4096] uncond
            uncond_mask:     [1, 1, 1, N_text]
            num_segments:    override config default
            seed:            base seed; each segment uses seed + segment_idx
            on_segment_done: optional callback `(segment_idx, segment_frames_np)`
                             — useful for the CLI to write `output_long_video_<i>.mp4`
                             intermediates as the run progresses (matches
                             upstream demo behavior).

        Returns: `[T_total, H, W, 3]` uint8 — concatenated frames.
        """
        num_segments = num_segments or self.config.num_segments
        cfg = self.config

        all_frames = []

        # --- Segment 1: T2V seed clip ---
        t2v_video = self.t2v(
            text_embeds=text_embeds, text_mask=text_mask,
            uncond_embeds=uncond_embeds, uncond_mask=uncond_mask,
            num_frames=cfg.num_frames_per_segment,
            height=cfg.height, width=cfg.width,
            seed=seed,
        )
        mx.eval(t2v_video)
        seg0_np = _mx_video_to_np_uint8(t2v_video)   # [T_seg, H, W, 3]
        all_frames.append(seg0_np)
        if on_segment_done:
            on_segment_done(0, seg0_np)
        current_np = seg0_np

        # --- Segments 2..N: chained Continuation ---
        for seg in range(1, num_segments):
            # The last `num_cond_frames` of the previous segment is the cond prefix
            prefix_np = current_np[-cfg.num_cond_frames:]
            prefix_mx = _np_uint8_to_mx_video(prefix_np)

            next_video = self.continuation(
                prefix_video=prefix_mx,
                text_embeds=text_embeds, text_mask=text_mask,
                uncond_embeds=uncond_embeds, uncond_mask=uncond_mask,
                num_new_frames=cfg.num_frames_per_segment - cfg.num_cond_frames,
                height=cfg.height, width=cfg.width,
                seed=seed + seg,
                return_full_video=False,   # strip the cond prefix from decoded output
            )
            mx.eval(next_video)
            seg_np = _mx_video_to_np_uint8(next_video)  # already prefix-stripped
            all_frames.append(seg_np)
            if on_segment_done:
                on_segment_done(seg, seg_np)
            # The current_np used for the NEXT segment's conditioning needs to
            # be the full last-segment-as-seen, including the conditioning
            # prefix it was generated against — concatenate to get continuity:
            current_np = np.concatenate([prefix_np, seg_np], axis=0)

        return np.concatenate(all_frames, axis=0)
