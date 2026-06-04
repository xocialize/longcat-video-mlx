"""MLX port of LongCat-Video — the BASE 13.6B video diffusion model.

Reference: https://github.com/meituan-longcat/LongCat-Video
PyTorch source: refs/longcat-video/longcat_video/

Six task variants share a single 48-block DiT checkpoint with optional LoRAs:
- T2V (text-to-video) — pipeline_t2v
- I2V (image-to-video) — pipeline_i2v
- Video Continuation — pipeline_continuation
- Long-Video — pipeline_long_video (chained Continuation)
- Interactive Video — pipeline_interactive (per-segment prompts)

Two LoRAs:
- cfg_step_lora: collapses CFG branches + reduces sampler step count
- refinement_lora: enables the 720p/30fps refinement pass (coarse-to-fine)

Companion repo `longcat-avatar-mlx` ports the Avatar 1.5 variant: same DiT
topology + audio cross-attention + Reference Skip + AudioProjModel. This
package vendors the shared modules (autoencoder_kl_wan, umt5, longcat_video_dit,
attention, blocks, rope_3d, lora) for self-contained operation.
"""

__version__ = "0.1.0.dev0"
