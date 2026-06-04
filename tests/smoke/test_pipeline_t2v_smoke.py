"""Smoke tests for pipeline_t2v.py — construction-time only, no weights."""

from __future__ import annotations

from longcat_video.pipeline_t2v import T2VPipelineConfig


def test_config_defaults_match_base_model():
    """Verify the T2V config defaults match the published base model config."""
    cfg = T2VPipelineConfig()
    assert cfg.scheduler_shift == 12.0, "base shift is 12.0 (Avatar's 7.0 is wrong)"
    assert cfg.num_sampling_steps == 50, "50-step baseline (no LoRA merge)"
    assert cfg.text_guidance_scale == 5.0
    assert cfg.cfg_collapse is False
    assert cfg.dit_in_channels == 16
    assert cfg.dit_out_channels == 16
    assert cfg.vae_scale_temporal == 4
    assert cfg.vae_scale_spatial == 8


def test_config_cfg_collapse_mode():
    """When cfg_step_lora is merged, caller flips cfg_collapse=True + scale=0."""
    cfg = T2VPipelineConfig(cfg_collapse=True, text_guidance_scale=0.0, num_sampling_steps=8)
    assert cfg.cfg_collapse is True
    assert cfg.text_guidance_scale == 0.0
    assert cfg.num_sampling_steps == 8


def test_pipeline_import_chain():
    """End-to-end import of all pipeline components without errors."""
    from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline
    from longcat_video.guidance import cfg_combine, flip_velocity_for_scheduler
    from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.models.umt5 import UMT5EncoderModel
    assert LongCatVideoT2VPipeline is not None


def test_cfg_collapse_normalizes_scalar_timestep():
    """Regression: cfg_collapse=True path used to skip the ndim==0 → [B=1]
    normalization that the 2-pass branch does. Scheduler returns 0-d
    timestep arrays, which then trickled through to the DiT and made
    `timestep.flatten()` collapse to (1,) instead of (B*N_t,) — silently
    corrupting the t_embedder output (256 instead of 512). Real-weights
    crash signature: `Last dimension of first input with shape (1, T, 256)
    must match second to last dimension of second input with shape (512,
    24576)` from inside `adaLN_modulation[1]`.

    Verifies the normalization by stubbing the DiT and asserting it
    receives a 1-D timestep, not a scalar.
    """
    import mlx.core as mx

    from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline, T2VPipelineConfig

    cfg = T2VPipelineConfig(cfg_collapse=True, num_sampling_steps=1)
    received_ndim = []

    class StubDiT:
        def __call__(self, lat, t, *a, **kw):
            received_ndim.append(int(t.ndim))
            return mx.zeros_like(lat)

    pipe = LongCatVideoT2VPipeline(
        vae=None, text_encoder=None, dit=StubDiT(),
        config=cfg, scheduler=None,
    )
    # 0-d (scalar) timestep — exactly what mlx-arsenal's scheduler emits
    t_scalar = mx.array(500.0)
    text_cat = mx.zeros((2, 1, 16, 4096))
    mask_cat = mx.ones((2, 16))
    pipe._cfg_forward(
        latents=mx.zeros((1, 16, 2, 4, 4)),
        timestep=t_scalar,
        text_embeds_cat=text_cat,
        text_mask_cat=mask_cat,
        uncond_text_embeds=text_cat[:1],
        uncond_text_mask=mask_cat[:1],
    )
    assert received_ndim == [1], (
        f"DiT should receive timestep with ndim=1, got ndim={received_ndim[0]}"
    )
