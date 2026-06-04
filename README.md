# longcat-video-mlx

Apple MLX port of [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's 13.6 B-parameter video diffusion model — for inference on Apple
Silicon (M-series).

> **Status: alpha — bf16 weights published.** Converted bf16 weights live
> at [mlx-community/LongCat-Video-bf16](https://huggingface.co/mlx-community/LongCat-Video-bf16)
> (42 GB). Part of the [LongCat-Video — MLX](https://huggingface.co/collections/mlx-community/longcat-video-mlx-6a216a3576c098e83c1cc167)
> collection. End-to-end T2V inference verified on real weights;
> refinement (720p) + LoRA merge wiring remaining for a polished release.
>
> Companion repo [xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
> ports the Avatar 1.5 variant of the same architecture — start there
> if you want audio-driven video generation.

## Six task variants, one DiT checkpoint

| Variant | Pipeline | Status |
|---|---|---|
| **T2V** — text-to-video | `pipeline_t2v` | ✅ shipped (B1.3 + B1.4 + golden smoke) |
| **I2V** — image-to-video | `pipeline_i2v` | ✅ shipped (B2.1) |
| **Video Continuation** | `pipeline_continuation` | ✅ shipped (B2.2) |
| **720p / 30fps refinement** | `refinement.py` (+ BSA) | ✅ shipped (B3.1 + B3.2 Tier A) |
| **Long-Video** (chained continuation) | `pipeline_long_video` | Pending (B5.1) |
| **Interactive Video** (per-segment prompts) | `pipeline_interactive` | Pending (B5.2) |
| Streamlit UI | — | Out of scope (use CLI) |

All six are driven by the same 13.6B DiT with optional LoRAs:
- `cfg_step_lora` collapses CFG branches + reduces sampler step count
- `refinement_lora` enables the 720p / 30fps refinement pass

## Reuse from the Avatar port

The Avatar port ([xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx))
already ships the shared architectural components — Wan VAE, umT5-XXL, base
DiT, attention primitives, blocks, 3D RoPE, LoRA loader. This repo
**copy-vendors** those modules (no cross-repo dependency at runtime) so it
can be developed and released independently.

## What's planned

See [LongCat-Video-Base-MLX-Port-Plan-v1.md](../XDocs/LongCat-Video-Base-MLX-Port-Plan-v1.md)
for the 5-week port plan. Stages:

- **B0** — Scaffold (this commit)
- **B1** — Convert base DiT + LoRAs, T2V pipeline + CLI, publish bf16
- **B2** — I2V + Continuation pipelines
- **B3** — Coarse-to-fine refinement + Block Sparse Attention (Tier A)
- **B4** — Block Sparse Attention (Tier B Metal kernel)
- **B5** — Long-Video + Interactive + MOS-style validation + q4/q8 publish

## License

MIT. Matches upstream Meituan LongCat-Video and the Avatar companion port.
