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
