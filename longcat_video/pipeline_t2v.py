"""Text-to-Video (T2V) inference pipeline for LongCat-Video base.

Mirrors the structure of Avatar's `pipeline_mlx.py` but stripped to the
text-only path:

- No audio (no Whisper, no AudioProjModel, no audio cross-attn)
- No reference image — pure noise latent at the head
- 2-pass text-only CFG (vs Avatar's 3-pass disentangled CFG)
- Standard scheduler shift=12.0 (vs Avatar's 7.0)
- Sampling steps from upstream defaults; can be reduced when
  `cfg_step_lora` is merged into the DiT

Two modes:

1. **Baseline** — 50-step Flow Matching with text-CFG (scale=5.0). The
   `cfg_step_lora` is NOT merged. Two DiT forward passes per step
   (cond + uncond). Slower but matches the published reference behavior.

2. **Fast (cfg_step_lora merged)** — caller pre-merges `cfg_step_lora`
   into the DiT via `merge_lora_into_model`. Then the pipeline can use
   `cfg_collapse=True` to skip the uncond pass (single DiT forward per
   step) and reduce step count (recipe-determined; ~8-16 steps typical
   for distilled CFG-step LoRAs).

For the refinement / 720p pass, see `refinement.py` (B3.1) — coarse
output here is the input to that pipeline when high-res is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from longcat_video.guidance import cfg_combine, cfg_split_outputs, flip_velocity_for_scheduler
from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.models.umt5 import UMT5EncoderModel


@dataclass
class T2VPipelineConfig:
    """Resolved config for a T2V pipeline instance."""

    # DiT
    dit_in_channels: int = 16
    dit_out_channels: int = 16

    # Sampler
    num_sampling_steps: int = 50          # baseline; reduce when cfg_step_lora merged
    num_train_timesteps: int = 1000
    scheduler_shift: float = 12.0          # base default; Avatar was 7.0

    # CFG
    text_guidance_scale: float = 5.0       # set 0.0 when cfg_step_lora is merged
    cfg_collapse: bool = False             # True = single DiT pass when cfg_step_lora merged

    # Video — coarse pass defaults
    num_frames: int = 24                   # 24 frames @ 15 fps ≈ 1.6 s coarse output
    target_fps: int = 15

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


class LongCatVideoT2VPipeline:
    """Composite T2V inference pipeline.

    Construct with already-loaded components:
        pipeline = LongCatVideoT2VPipeline(
            vae=vae, text_encoder=umt5, dit=base_dit, config=T2VPipelineConfig()
        )
        video = pipeline(
            text_embeds=text_embeds,          # [1, N_text, 4096]
            text_mask=text_mask,              # [1, N_text]
            uncond_embeds=uncond_embeds,      # [1, N_text, 4096]
            uncond_mask=uncond_mask,
            seed=0,
        )

    For end-to-end with raw prompts, see `scripts/run_t2v.py` (B1.4) which
    handles tokenization, image preprocessing (N/A here), and umT5 encoding.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        dit: LongCatVideoTransformer3DModel,
        config: Optional[T2VPipelineConfig] = None,
        scheduler=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.dit = dit
        self.config = config or T2VPipelineConfig()

        if scheduler is None:
            try:
                from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler
                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.config.num_train_timesteps,
                    shift=self.config.scheduler_shift,
                )
            except ImportError as e:
                raise ImportError(
                    "mlx-arsenal is required for the default scheduler — "
                    "`pip install mlx-arsenal`"
                ) from e
        self.scheduler = scheduler

    # ----- Helpers --------------------------------------------------------

    def _make_initial_noise(
        self, batch_size: int, num_frames: int, height: int, width: int, seed: int
    ) -> mx.array:
        """Sample a Gaussian noise tensor in latent space.

        Latent shape: `[B, in_channels=16, T_lat, H_lat, W_lat]` where
        `T_lat = 1 + (num_frames - 1) // vae_scale_temporal` and
        `H_lat = height // vae_scale_spatial, W_lat = width // vae_scale_spatial`.

        For T2V there's NO reference frame — the entire latent is noise.
        """
        v = self.config.vae_scale_temporal
        s = self.config.vae_scale_spatial
        T_lat = 1 + (num_frames - 1) // v
        H_lat = height // s
        W_lat = width // s
        mx.random.seed(seed)
        return mx.random.normal(
            (batch_size, self.config.dit_in_channels, T_lat, H_lat, W_lat)
        )

    def _cfg_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds_cat: mx.array,
        text_mask_cat: mx.array,
        uncond_text_embeds: mx.array,
        uncond_text_mask: mx.array,
    ) -> mx.array:
        """One CFG step: 2-pass forward (cond + uncond) + combine + flip.

        When `cfg_collapse=True` (cfg_step_lora merged), runs only the
        cond pass and skips the (cond - uncond) correction.
        """
        if self.config.cfg_collapse:
            # cfg_step_lora has absorbed the CFG correction — single pass.
            # Normalize scalar timestep to [B=1] just like the 2-pass branch
            # does, otherwise the DiT's `timestep.ndim == 1` broadcast check
            # is skipped and downstream reshape mangles the embedding dim.
            if timestep.ndim == 0:
                timestep = timestep[None]
            pred = self.dit(
                latents, timestep, text_embeds_cat[1:2],  # positive half only
                encoder_attention_mask=text_mask_cat[1:2],
                num_cond_latents=0,
            )
            return flip_velocity_for_scheduler(pred)

        # Standard 2-pass CFG
        latents_2 = mx.concatenate([latents, latents], axis=0)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep_2 = mx.repeat(timestep, 2, axis=0)
        pred_2 = self.dit(
            latents_2, timestep_2, text_embeds_cat,
            encoder_attention_mask=text_mask_cat,
            num_cond_latents=0,
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
        """Run the full T2V denoising loop and return decoded video.

        Args:
            text_embeds:     [1, 1, N_text, 4096] from umT5 (caller wraps the
                             extra singleton dim — matches the DiT's expected
                             shape, same as Avatar).
            text_mask:       [1, N_text] valid-token mask
            uncond_embeds:   [1, 1, N_text, 4096] for the empty/uncond prompt
            uncond_mask:     [1, N_text]
            initial_noise:   optional [B, 16, T_lat, H_lat, W_lat]; bypasses
                             the seeded MLXRandom path (needed for Python ↔
                             Swift parity since MLX random isn't seed-compat).

        Returns:
            Video tensor `[1, 3, num_frames, H_out, W_out]` in `[-1, 1]`.
        """
        num_frames = num_frames or self.config.num_frames

        # 1. Initial noise (or caller-provided for parity)
        if initial_noise is not None:
            latents = initial_noise
        else:
            latents = self._make_initial_noise(1, num_frames, height, width, seed)

        # 2. Set up scheduler
        self.scheduler.set_timesteps(self.config.num_sampling_steps)
        timesteps = self.scheduler.timesteps

        # 3. Stack uncond + cond text for batched CFG pass
        text_embeds_cat = mx.concatenate([uncond_embeds, text_embeds], axis=0)
        text_mask_cat = mx.concatenate([uncond_mask, text_mask], axis=0)

        # 4. Denoising loop
        for i, t in enumerate(timesteps):
            t_arr = mx.array([t], dtype=mx.float32) if not isinstance(t, mx.array) else t
            noise_pred = self._cfg_forward(
                latents, t_arr,
                text_embeds_cat, text_mask_cat,
                uncond_embeds, uncond_mask,
            )
            # NOTE: mlx-arsenal scheduler.step returns mx.array directly
            # (not the diffusers (sample, ...) tuple). No [0] indexing.
            latents = self.scheduler.step(noise_pred, t, latents)

        # 5. Decode through VAE (denormalize first → match PT _decode convention)
        z_denorm = self.vae.denormalize_latents(latents)
        video = self.vae.decode(z_denorm)
        return video
