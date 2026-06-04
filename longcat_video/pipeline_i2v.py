"""Image-to-Video (I2V) inference pipeline for LongCat-Video base.

Same DiT + scheduler + CFG as T2V, but conditions on a single reference
frame at the head of the temporal axis:

    latents = concat([first_frame_latent, noise_latent], axis=time)
    num_cond_latents = 1

The reference frame is a **motion anchor** (the video starts from it),
NOT a persistent identity. This is the key semantic difference from
Avatar's reference image (which the Reference Skip mechanism kept the
DiT from copying into every frame).

The DiT's existing `num_cond_latents` parameter handles the conditioning
branching internally — no DiT changes needed.

For continuation (last-N-frames-as-cond), see pipeline_continuation.py.
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
class I2VPipelineConfig:
    """I2V config — superset of T2V config with image conditioning."""

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
    num_frames: int = 24
    target_fps: int = 15

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


class LongCatVideoI2VPipeline:
    """I2V inference pipeline. Same DiT/CFG as T2V; conditions on a single
    reference frame at the head of the temporal axis.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[I2VPipelineConfig] = None,
        scheduler=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or I2VPipelineConfig()

        if scheduler is None:
            from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=self.config.scheduler_shift,
            )
        self.scheduler = scheduler

    # ----- Helpers --------------------------------------------------------

    def _encode_reference_image(self, image: mx.array) -> mx.array:
        """`image`: `[B=1, 3, T=1, H, W]` in `[-1, 1]` (single-frame video).
        Returns: `[B, z_dim=16, 1, H_lat, W_lat]` normalized latent.
        """
        raw_mu = self.vae.encode(image)
        return self.vae.normalize_latents(raw_mu)

    def _make_noise(
        self, num_frames: int, height: int, width: int, seed: int
    ) -> mx.array:
        v = self.config.vae_scale_temporal
        s = self.config.vae_scale_spatial
        T_lat = 1 + (num_frames - 1) // v
        H_lat = height // s
        W_lat = width // s
        mx.random.seed(seed)
        return mx.random.normal((1, self.config.dit_in_channels, T_lat, H_lat, W_lat))

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
        image: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_frames: Optional[int] = None,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
        initial_noise: Optional[mx.array] = None,
    ) -> mx.array:
        """Run the I2V denoising loop.

        Args:
            image:           [1, 3, 1, H, W] in [-1, 1] — single reference frame
            text_embeds:     [1, 1, N_text, 4096]
            text_mask:       [1, N_text]
            uncond_embeds:   [1, 1, N_text, 4096]
            uncond_mask:     [1, N_text]
            initial_noise:   optional caller-provided noise (parity testing)

        Returns: `[1, 3, num_frames, H_out, W_out]` in `[-1, 1]`.
        """
        num_frames = num_frames or self.config.num_frames

        # 1. Encode reference image
        ref_latent = self._encode_reference_image(image)   # [1, 16, 1, H_lat, W_lat]
        num_cond_latents = 1

        # 2. Initial noise + concat ref
        if initial_noise is not None:
            noise = initial_noise
        else:
            noise = self._make_noise(num_frames, height, width, seed)
        latents = mx.concatenate([ref_latent, noise], axis=2)

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

        # 6. Strip ref latent + decode
        denoised = latents[:, :, num_cond_latents:]
        z_denorm = self.vae.denormalize_latents(denoised)
        return self.vae.decode(z_denorm)
