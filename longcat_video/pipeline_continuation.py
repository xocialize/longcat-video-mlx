"""Video Continuation inference pipeline for LongCat-Video base.

Same DiT + scheduler + CFG as T2V, but conditions on the **last N frames**
of a prior video clip at the head of the temporal axis:

    cond_latents = vae.encode(last_N_frames)             # [1, 16, T_cond_lat, H_lat, W_lat]
    latents      = concat([cond_latents, noise], axis=time)
    num_cond_latents = T_cond_lat

Compared to I2V (which uses a single reference frame at T=0), continuation
provides a multi-frame motion prefix that the DiT extends. The chunked-
attention path inside the DiT — which Avatar used to keep attention sane
across very long videos — drops in directly here with audio cross-attn
disabled.

For coarse-to-fine refinement (480p → 720p), see `refinement.py` (B3.1) —
continuation output is the input to that pipeline when high-res is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from longcat_video.guidance import cfg_combine, cfg_split_outputs, flip_velocity_for_scheduler
from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.models.umt5 import UMT5EncoderModel


@dataclass
class ContinuationPipelineConfig:
    """Video-continuation config — like T2V plus a prefix-clip cond."""

    # DiT
    dit_in_channels: int = 16
    dit_out_channels: int = 16

    # Sampler
    num_sampling_steps: int = 50
    num_train_timesteps: int = 1000
    scheduler_shift: float = 12.0

    # CFG
    text_guidance_scale: float = 5.0
    cfg_collapse: bool = False

    # Video
    num_new_frames: int = 24        # how many frames to GENERATE (excluding the cond prefix)
    target_fps: int = 15

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


class LongCatVideoContinuationPipeline:
    """Continuation inference pipeline. Conditions on the trailing frames
    of a prior video clip and extends.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[ContinuationPipelineConfig] = None,
        scheduler=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or ContinuationPipelineConfig()

        if scheduler is None:
            from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=self.config.scheduler_shift,
            )
        self.scheduler = scheduler

    # ----- Helpers --------------------------------------------------------

    def _encode_prefix_clip(self, prefix_video: mx.array) -> mx.array:
        """`prefix_video`: `[1, 3, T_cond, H, W]` in `[-1, 1]` — the trailing
        frames of the prior clip to condition on.

        Returns: `[1, 16, T_cond_lat, H_lat, W_lat]` normalized latent.

        The number of temporal latent frames is determined by the VAE's
        downsampling: `T_cond_lat = 1 + (T_cond - 1) // vae_scale_temporal`.
        Callers typically want T_cond such that this comes out to ≥1 latent
        frame (so at least `vae_scale_temporal + 1 = 5` raw frames).
        """
        raw_mu = self.vae.encode(prefix_video)
        return self.vae.normalize_latents(raw_mu)

    def _make_noise(
        self, num_new_frames: int, height: int, width: int, seed: int
    ) -> mx.array:
        """Sample noise for the NEW frames only — the cond prefix is
        provided separately and concatenated in `__call__`.
        """
        v = self.config.vae_scale_temporal
        s = self.config.vae_scale_spatial
        T_new_lat = 1 + (num_new_frames - 1) // v
        H_lat = height // s
        W_lat = width // s
        mx.random.seed(seed)
        return mx.random.normal(
            (1, self.config.dit_in_channels, T_new_lat, H_lat, W_lat)
        )

    def _cfg_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds_cat: mx.array,
        text_mask_cat: mx.array,
        num_cond_latents: int,
    ) -> mx.array:
        """One CFG step with conditioning latents at the head."""
        if self.config.cfg_collapse:
            # See pipeline_t2v._cfg_forward — same ndim==0 normalization.
            if timestep.ndim == 0:
                timestep = timestep[None]
            pred = self.dit(
                latents, timestep, text_embeds_cat[1:2],
                encoder_attention_mask=text_mask_cat[1:2],
                num_cond_latents=num_cond_latents,
            )
            return flip_velocity_for_scheduler(pred)

        latents_2 = mx.concatenate([latents, latents], axis=0)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep_2 = mx.repeat(timestep, 2, axis=0)
        pred_2 = self.dit(
            latents_2, timestep_2, text_embeds_cat,
            encoder_attention_mask=text_mask_cat,
            num_cond_latents=num_cond_latents,
        )
        noise_uncond, noise_cond = cfg_split_outputs(pred_2)
        combined = cfg_combine(
            noise_cond, noise_uncond,
            text_guidance_scale=self.config.text_guidance_scale,
        )
        return flip_velocity_for_scheduler(combined)

    # ----- Inference ------------------------------------------------------

    def __call__(
        self,
        prefix_video: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_new_frames: Optional[int] = None,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
        initial_noise: Optional[mx.array] = None,
        return_full_video: bool = True,
    ) -> mx.array:
        """Run the continuation denoising loop.

        Args:
            prefix_video:    [1, 3, T_cond, H, W] in [-1, 1] — prior clip's
                             trailing frames (motion + content prefix)
            text_embeds:     [1, 1, N_text, 4096]
            text_mask:       [1, N_text]
            uncond_embeds:   [1, 1, N_text, 4096]
            uncond_mask:     [1, N_text]
            initial_noise:   optional caller-provided noise (parity testing)
            return_full_video: if True, decoded output includes the cond
                               prefix (T = T_cond + T_new); if False, just
                               the newly generated tail (T = T_new).

        Returns: video tensor in `[-1, 1]`, layout `[1, 3, T, H_out, W_out]`.
        """
        num_new_frames = num_new_frames or self.config.num_new_frames

        # 1. Encode the prefix clip
        cond_latent = self._encode_prefix_clip(prefix_video)
        num_cond_latents = int(cond_latent.shape[2])

        # 2. Initial noise for NEW frames + concat at head
        if initial_noise is not None:
            noise = initial_noise
        else:
            noise = self._make_noise(num_new_frames, height, width, seed)
        latents = mx.concatenate([cond_latent, noise], axis=2)

        # 3. Scheduler
        self.scheduler.set_timesteps(self.config.num_sampling_steps)
        timesteps = self.scheduler.timesteps

        # 4. Stack CFG text
        text_embeds_cat = mx.concatenate([uncond_embeds, text_embeds], axis=0)
        text_mask_cat = mx.concatenate([uncond_mask, text_mask], axis=0)

        # 5. Denoise
        for i, t in enumerate(timesteps):
            t_arr = mx.array([t], dtype=mx.float32) if not isinstance(t, mx.array) else t
            noise_pred = self._cfg_forward(
                latents, t_arr, text_embeds_cat, text_mask_cat,
                num_cond_latents=num_cond_latents,
            )
            latents = self.scheduler.step(noise_pred, t, latents)

        # 6. Optional: strip cond prefix from output
        if not return_full_video:
            latents = latents[:, :, num_cond_latents:]

        # 7. Decode
        z_denorm = self.vae.denormalize_latents(latents)
        return self.vae.decode(z_denorm)
