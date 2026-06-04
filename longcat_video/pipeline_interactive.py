"""Interactive Video orchestration: per-segment prompts.

Same chaining shape as Long-Video (`pipeline_long_video.py`) — T2V seed
+ chained Continuation — but each segment can take a *different* prompt.
This is the building block for interactive / dialogue-driven video
generation where the user's prompt evolves between segments:

  segment 1: "A cat enters the frame, looking around curiously"
  segment 2: "The cat sees a butterfly and starts chasing it"
  segment 3: "The butterfly leads the cat to a sunny meadow"
  ...

Encoded prompt list is supplied by the caller (caller is responsible for
tokenizing each prompt via umT5 — see `scripts/run_interactive.py`).

The smoothness of the prompt transitions depends entirely on:
- How semantically close adjacent prompts are
- The `num_cond_frames` budget (more cond frames = smoother but slower
  semantic change)

For one-prompt-fits-all use Long-Video instead — same code path with
the prompt list collapsed to a single embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import mlx.core as mx
import numpy as np

from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.models.umt5 import UMT5EncoderModel
from longcat_video.pipeline_continuation import (
    ContinuationPipelineConfig,
    LongCatVideoContinuationPipeline,
)
from longcat_video.pipeline_long_video import (
    _mx_video_to_np_uint8,
    _np_uint8_to_mx_video,
)
from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline, T2VPipelineConfig


@dataclass
class InteractivePipelineConfig:
    """Config for Interactive Video orchestration."""

    # Per-segment defaults
    num_frames_per_segment: int = 93
    num_cond_frames: int = 13

    # Resolution
    height: int = 480
    width: int = 832
    target_fps: int = 15

    # Per-pipeline settings
    num_sampling_steps: int = 50
    text_guidance_scale: float = 5.0
    scheduler_shift: float = 12.0
    cfg_collapse: bool = False

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


class LongCatVideoInteractivePipeline:
    """Interactive Video pipeline = per-segment-prompt chained generation.

    `__call__` takes a `Sequence` of pre-encoded per-segment prompts
    (`[text_embeds, text_mask, uncond_embeds, uncond_mask]` 4-tuples).
    Length of the sequence = number of segments to generate.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[InteractivePipelineConfig] = None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or InteractivePipelineConfig()

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
        prompts_encoded: Sequence[tuple[mx.array, mx.array, mx.array, mx.array]],
        seed: int = 0,
        on_segment_done: Optional[
            Callable[[int, str, np.ndarray], None]
        ] = None,
        prompt_labels: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Run the interactive-video chain.

        Args:
            prompts_encoded: sequence of 4-tuples
                `(text_embeds, text_mask, uncond_embeds, uncond_mask)`
                one per segment. text_embeds: `[1, 1, N_text, 4096]`,
                text_mask: `[1, 1, 1, N_text]` (the DiT-broadcast form).
            seed: base seed; each segment uses `seed + segment_idx`.
            on_segment_done: optional callback
                `(segment_idx, prompt_label, segment_frames_np)`.
                `prompt_label` is from `prompt_labels[i]` if provided,
                otherwise `f"segment {i}"`.
            prompt_labels: optional human-readable labels for the
                callback / debugging (e.g. the raw prompt strings).

        Returns: `[T_total, H, W, 3]` uint8 — concatenated frames.

        Raises:
            ValueError: if `prompts_encoded` is empty.
        """
        if not prompts_encoded:
            raise ValueError("prompts_encoded must contain at least 1 prompt")
        cfg = self.config

        all_frames = []

        labels = prompt_labels or [f"segment {i}" for i in range(len(prompts_encoded))]

        # --- Segment 0: T2V seed clip with the first prompt ---
        te, tm, ue, um = prompts_encoded[0]
        t2v_video = self.t2v(
            text_embeds=te, text_mask=tm,
            uncond_embeds=ue, uncond_mask=um,
            num_frames=cfg.num_frames_per_segment,
            height=cfg.height, width=cfg.width,
            seed=seed,
        )
        mx.eval(t2v_video)
        seg0_np = _mx_video_to_np_uint8(t2v_video)
        all_frames.append(seg0_np)
        if on_segment_done:
            on_segment_done(0, labels[0], seg0_np)
        current_np = seg0_np

        # --- Segments 1..N-1: per-prompt Continuation ---
        for seg in range(1, len(prompts_encoded)):
            te, tm, ue, um = prompts_encoded[seg]
            prefix_np = current_np[-cfg.num_cond_frames:]
            prefix_mx = _np_uint8_to_mx_video(prefix_np)

            next_video = self.continuation(
                prefix_video=prefix_mx,
                text_embeds=te, text_mask=tm,
                uncond_embeds=ue, uncond_mask=um,
                num_new_frames=cfg.num_frames_per_segment - cfg.num_cond_frames,
                height=cfg.height, width=cfg.width,
                seed=seed + seg,
                return_full_video=False,
            )
            mx.eval(next_video)
            seg_np = _mx_video_to_np_uint8(next_video)
            all_frames.append(seg_np)
            if on_segment_done:
                on_segment_done(seg, labels[seg], seg_np)
            current_np = np.concatenate([prefix_np, seg_np], axis=0)

        return np.concatenate(all_frames, axis=0)
