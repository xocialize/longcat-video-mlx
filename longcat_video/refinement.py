"""Coarse-to-fine refinement pipeline for LongCat-Video base.

Second-pass super-resolution + (optional) temporal upsampling on top of a
coarse stage-1 video. This is the "720p / 30fps refinement" pass that
turns a 480p/15fps coarse output into a 720p/30fps refined output.

Architecturally it's **SDEdit-style** (a.k.a. img2img / partial-noise
denoising), NOT a fresh sampling from pure noise:

    stage1_video (low-res np frames)
        ↓ trilinear upsample to (T_new, 720H, 720W) with T_new = 2*T_old
        ↓ pad front/back with end-frame replication for BSA chunk alignment
        ↓ VAE encode → latent_up
        ↓ partial-noise:   z = (1-τ) * latent_up + τ * randn       (τ = t_thresh = 0.5)
        ↓ denoising loop, ONLY over timesteps ≤ τ*1000 (so half the schedule)
        ↓ VAE decode + slice front padding off
    refined_video (high-res np frames)

Differences from T2V/I2V/Continuation pipelines:

- **No CFG** — the refinement_lora was trained to produce final outputs at
  guidance_scale=1.0; one DiT forward per step. This is what makes 50
  refinement steps roughly equivalent in wall-time to 25 baseline T2V
  steps (which need 2 DiT calls each for CFG).
- **Timestep schedule truncated** to ≤ t_thresh*1000 — so 50 nominal
  steps become ~25 effective steps (denoising starts halfway down the
  schedule from the partial-noise injection point).
- **Cond latents are frozen** — `timestep[:, :num_cond_latents] = 0`
  inside the DiT call, and `scheduler.step` only updates the noise slice.
  When the upstream `--num-cond-frames > 0`, the front of the latent
  tensor is the cond clip held at t=0 the whole loop.
- **BSA enabled in the DiT** — caller is responsible for flipping
  `dit.enable_bsa=True` and hot-swapping `refinement_lora` before
  constructing this pipeline. BSA is what makes 720p attention tractable
  (see `docs/development/bsa-config-found.md`).
- **Padding for BSA chunk alignment** — `num_noise_latents` is rounded
  up to a multiple of 4 (chunk_3d_shape_q[0]=4) by replicating the last
  frame; padding is sliced off after decode.

For weights / LoRA hot-swap, see `scripts/run_refine.py` (B3.1 CLI) —
this module only handles the pipeline mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from longcat_video.guidance import flip_velocity_for_scheduler
from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.models.umt5 import UMT5EncoderModel


@dataclass
class RefinementPipelineConfig:
    """Refinement-pass config."""

    # DiT
    dit_in_channels: int = 16
    dit_out_channels: int = 16

    # Sampler
    num_sampling_steps: int = 50         # nominal; t_thresh truncates the schedule
    num_train_timesteps: int = 1000
    scheduler_shift: float = 12.0
    t_thresh: float = 0.5                # SDEdit injection point (fraction of [0, 1])

    # Resolution targets
    target_height: int = 720             # 720p coarse-to-fine default
    target_width: int = 1280
    spatial_refine_only: bool = False    # True keeps frame count; False doubles it (15→30fps)

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8
    bsa_latent_granularity: int = 4      # chunk_3d_shape_q[0]; latents padded to multiple of this


def _bilinear_resize_5d(
    x: mx.array, new_T: int, new_H: int, new_W: int, align_corners: bool = True,
) -> mx.array:
    """5-D trilinear upsample: `[B, C, T, H, W]` → `[B, C, new_T, new_H, new_W]`.

    Implemented as separable linear-time + bilinear-space using
    `mx.array` indexing (no MLX-native trilinear yet). align_corners=True
    matches the PT reference.

    Done in float32 for stability — the input slice usually arrives in
    bf16, but resampling in low precision spuriously washes detail.
    """
    B, C, T, H, W = x.shape
    x_f = x.astype(mx.float32)

    # --- temporal linear ---
    if new_T != T:
        if T == 1:
            x_t = mx.repeat(x_f, new_T, axis=2)
        else:
            if align_corners:
                # t_src ∈ [0, T-1], evenly spaced new_T samples
                ts = mx.linspace(0.0, T - 1, num=new_T).astype(mx.float32)
            else:
                # half-pixel centers
                ts = (mx.arange(new_T).astype(mx.float32) + 0.5) * (T / new_T) - 0.5
                ts = mx.clip(ts, 0.0, T - 1)
            t0 = mx.floor(ts).astype(mx.int32)
            t1 = mx.minimum(t0 + 1, T - 1)
            wt = (ts - t0.astype(mx.float32))[None, None, :, None, None]
            a = mx.take(x_f, t0, axis=2)
            b = mx.take(x_f, t1, axis=2)
            x_t = a * (1.0 - wt) + b * wt
    else:
        x_t = x_f

    # --- spatial bilinear (H) ---
    if new_H != H:
        if H == 1:
            x_t = mx.repeat(x_t, new_H, axis=3)
        else:
            if align_corners:
                hs = mx.linspace(0.0, H - 1, num=new_H).astype(mx.float32)
            else:
                hs = (mx.arange(new_H).astype(mx.float32) + 0.5) * (H / new_H) - 0.5
                hs = mx.clip(hs, 0.0, H - 1)
            h0 = mx.floor(hs).astype(mx.int32)
            h1 = mx.minimum(h0 + 1, H - 1)
            wh = (hs - h0.astype(mx.float32))[None, None, None, :, None]
            a = mx.take(x_t, h0, axis=3)
            b = mx.take(x_t, h1, axis=3)
            x_t = a * (1.0 - wh) + b * wh

    # --- spatial bilinear (W) ---
    if new_W != W:
        if W == 1:
            x_t = mx.repeat(x_t, new_W, axis=4)
        else:
            if align_corners:
                ws = mx.linspace(0.0, W - 1, num=new_W).astype(mx.float32)
            else:
                ws = (mx.arange(new_W).astype(mx.float32) + 0.5) * (W / new_W) - 0.5
                ws = mx.clip(ws, 0.0, W - 1)
            w0 = mx.floor(ws).astype(mx.int32)
            w1 = mx.minimum(w0 + 1, W - 1)
            ww = (ws - w0.astype(mx.float32))[None, None, None, None, :]
            a = mx.take(x_t, w0, axis=4)
            b = mx.take(x_t, w1, axis=4)
            x_t = a * (1.0 - ww) + b * ww

    return x_t.astype(x.dtype)


def _pad_replicate_temporal(x: mx.array, pad_front: int, pad_back: int) -> mx.array:
    """Replicate first / last frame along the temporal axis (axis=2)."""
    parts = []
    if pad_front > 0:
        parts.append(mx.repeat(x[:, :, 0:1], pad_front, axis=2))
    parts.append(x)
    if pad_back > 0:
        parts.append(mx.repeat(x[:, :, -1:], pad_back, axis=2))
    return mx.concatenate(parts, axis=2)


def _compute_padding(
    num_cond_frames: int, num_noise_frames: int,
    vae_scale_temporal: int, bsa_latent_granularity: int,
) -> tuple[int, int, int, int]:
    """Compute padding so both cond and noise latent counts are multiples
    of `bsa_latent_granularity`. Mirrors upstream lines 1240-1255.

    Returns `(num_cond_latents, num_cond_frames_added,
              num_noise_latents, num_noise_frames_added)`.
    """
    num_cond_latents = 0
    num_cond_frames_added = 0
    if num_cond_frames > 0:
        num_cond_latents = 1 + math.ceil((num_cond_frames - 1) / vae_scale_temporal)
        num_cond_latents = (
            math.ceil(num_cond_latents / bsa_latent_granularity)
            * bsa_latent_granularity
        )
        num_cond_frames_added = (
            1 + (num_cond_latents - 1) * vae_scale_temporal - num_cond_frames
        )

    num_noise_latents = math.ceil(num_noise_frames / vae_scale_temporal)
    num_noise_latents = (
        math.ceil(num_noise_latents / bsa_latent_granularity)
        * bsa_latent_granularity
    )
    num_noise_frames_added = (
        num_noise_latents * vae_scale_temporal - num_noise_frames
    )
    return (
        num_cond_latents, num_cond_frames_added,
        num_noise_latents, num_noise_frames_added,
    )


class LongCatVideoRefinementPipeline:
    """Refinement (second-pass coarse-to-fine) inference pipeline.

    **Caller is responsible for:**
    1. Loading `refinement_lora` into the DiT before constructing this.
    2. Setting `dit.enable_bsa = True` (or the equivalent attribute the
       DiT exposes) so the second pass uses BSA for the 720p attention.

    This pipeline wraps the SDEdit-style upsample + partial-noise +
    truncated-schedule denoise + decode flow.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[RefinementPipelineConfig] = None,
        scheduler=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or RefinementPipelineConfig()

        if scheduler is None:
            from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=self.config.scheduler_shift,
            )
        self.scheduler = scheduler

    # ----- Helpers --------------------------------------------------------

    def _prepare_upsampled_latent(
        self,
        stage1_video_np: np.ndarray,
        num_cond_frames: int,
        seed: int,
    ) -> tuple[mx.array, int, int, int]:
        """Upsample, pad, encode → noisy latent ready for the denoise loop.

        Args:
            stage1_video_np: [T_old, H_old, W_old, 3] uint8 frames from
                             the coarse pass.
            num_cond_frames: how many leading frames of stage1_video are
                             treated as cond (frozen at t=0); 0 = none.

        Returns:
            (latent_noisy, num_cond_latents, num_cond_frames_added,
             new_frame_size)
        """
        cfg = self.config
        H, W = cfg.target_height, cfg.target_width

        # 1. stage1 → [1, 3, T_old, H_old, W_old] in [0, 1]
        s1 = np.asarray(stage1_video_np, dtype=np.float32) / 255.0  # [T, H, W, 3]
        s1 = s1.transpose(3, 0, 1, 2)[None, :]
        s1_mx = mx.array(s1)

        T_old = int(s1_mx.shape[2])
        new_T = T_old if cfg.spatial_refine_only else 2 * T_old

        # 2. Trilinear upsample to (new_T, H, W), then normalize to [-1, 1]
        up = _bilinear_resize_5d(s1_mx, new_T=new_T, new_H=H, new_W=W,
                                 align_corners=True)
        up = up * 2.0 - 1.0

        # 3. Padding for BSA chunk alignment
        num_noise_frames = int(up.shape[2]) - num_cond_frames
        (num_cond_latents, num_cond_frames_added,
         num_noise_latents, num_noise_frames_added) = _compute_padding(
            num_cond_frames=num_cond_frames,
            num_noise_frames=num_noise_frames,
            vae_scale_temporal=cfg.vae_scale_temporal,
            bsa_latent_granularity=cfg.bsa_latent_granularity,
        )
        up = _pad_replicate_temporal(
            up, pad_front=num_cond_frames_added, pad_back=num_noise_frames_added,
        )

        # 4. VAE encode + normalize latents
        raw_mu = self.vae.encode(up)
        latent_up = self.vae.normalize_latents(raw_mu)

        # 5. SDEdit-style partial-noise injection
        t_thresh = cfg.t_thresh
        mx.random.seed(seed)
        noise = mx.random.normal(latent_up.shape)
        latent_noisy = (1.0 - t_thresh) * latent_up + t_thresh * noise

        return latent_noisy, num_cond_latents, num_cond_frames_added, new_T

    def _truncate_timesteps(self, timesteps: mx.array) -> mx.array:
        """Clip the timestep schedule to `≤ t_thresh * 1000`.

        Inserts the threshold itself as the first timestep, matching
        upstream's `t_thresh_tensor.unsqueeze(0) + timesteps[timesteps < t_thresh]`.
        Schedulers seeded with a 50-step schedule end up running ~25 actual
        denoising steps from t=500 down.
        """
        thresh = self.config.t_thresh * 1000.0
        ts_np = np.asarray(timesteps).astype(np.float32)
        keep = ts_np[ts_np < thresh]
        return mx.array(np.concatenate([[thresh], keep]).astype(np.float32))

    # ----- Inference ------------------------------------------------------

    def __call__(
        self,
        stage1_video_np: np.ndarray,
        text_embeds: mx.array,
        text_mask: mx.array,
        num_cond_frames: int = 0,
        seed: int = 0,
    ) -> mx.array:
        """Run the refinement denoise loop.

        Args:
            stage1_video_np: [T_old, H_old, W_old, 3] uint8 coarse output
            text_embeds:     [1, 1, N_text, 4096]
            text_mask:       [1, N_text] (raw umT5 mask, not the broadcast form)
            num_cond_frames: leading frames of stage1 treated as cond
                             (frozen at t=0); 0 = pure refinement.

        Returns: `[1, 3, T_out, H, W]` in `[-1, 1]`, with the front
        padding sliced off so `T_out = new_frame_size + num_cond_frames`.
        """
        cfg = self.config

        # 1. Prepare upsampled + partial-noise latent
        latents, num_cond_latents, num_cond_frames_added, new_T = (
            self._prepare_upsampled_latent(
                stage1_video_np, num_cond_frames=num_cond_frames, seed=seed,
            )
        )

        # 2. Truncated timestep schedule
        self.scheduler.set_timesteps(cfg.num_sampling_steps)
        timesteps = self._truncate_timesteps(self.scheduler.timesteps)
        # tell the scheduler about the truncated schedule (mlx-arsenal supports this)
        if hasattr(self.scheduler, "timesteps"):
            self.scheduler.timesteps = timesteps

        # 3. Denoising loop — single forward per step, no CFG
        for i, t in enumerate(timesteps):
            t_scalar = float(t) if not isinstance(t, mx.array) else float(t.item())
            # Per-timestep + per-cond-latent mask: cond positions held at t=0
            ts_full = mx.full(
                (1, int(latents.shape[2])), t_scalar, dtype=mx.float32,
            )
            if num_cond_latents > 0:
                # zero-out the first num_cond_latents columns
                mask = mx.concatenate([
                    mx.zeros((1, num_cond_latents), dtype=mx.float32),
                    mx.ones((1, int(latents.shape[2]) - num_cond_latents),
                            dtype=mx.float32),
                ], axis=1)
                ts_full = ts_full * mask

            noise_pred_cond = self.dit(
                latents, ts_full, text_embeds,
                encoder_attention_mask=text_mask,
                num_cond_latents=num_cond_latents,
            )
            noise_pred = flip_velocity_for_scheduler(noise_pred_cond)

            # Step only the noise slice; cond stays frozen
            if num_cond_latents > 0:
                noisy_slice = latents[:, :, num_cond_latents:]
                step_in = noise_pred[:, :, num_cond_latents:]
                stepped = self.scheduler.step(step_in, t, noisy_slice)
                latents = mx.concatenate(
                    [latents[:, :, :num_cond_latents], stepped], axis=2,
                )
            else:
                latents = self.scheduler.step(noise_pred, t, latents)

        # 4. Decode
        z_denorm = self.vae.denormalize_latents(latents)
        video = self.vae.decode(z_denorm)

        # 5. Slice off the front padding so output starts at the
        # first stage1 frame (upstream: `output_video[:, num_cond_frames_added: new_frame_size+num_cond_frames_added]`)
        if num_cond_frames_added > 0:
            new_frame_size = new_T
            end = num_cond_frames_added + new_frame_size + num_cond_frames
            video = video[:, :, num_cond_frames_added:end]

        return video
