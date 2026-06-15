# longcat-video-mlx

Apple MLX port of [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's 13.6 B-parameter video diffusion model — for inference on Apple
Silicon (M-series).

> **Status: production-tier.** Three quantization variants live on HF — pick by RAM
> budget. Part of the
> [LongCat-Video — MLX](https://huggingface.co/collections/mlx-community/longcat-video-mlx-6a216a3576c098e83c1cc167) collection.
> All six task variants ship and verify; DiT block parity vs the PT reference is
> **8.94e-08 (fp32 precision noise)**; the 720p refinement pass has a Block Sparse
> Attention Metal-kernel ladder culminating in a `simdgroup_matrix` HW-accelerated
> kernel.
>
> Companion repo [xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
> ports the Avatar 1.5 variant of the same architecture — start there
> if you want audio-driven video generation.

## Three variants — pick by RAM budget

| Variant | DiT size | Total disk | Min RAM | Quality | HF |
|---|---|---|---|---|---|
| [**bf16**](https://huggingface.co/mlx-community/LongCat-Video-bf16) | 26 GB | 42 GB | 64 GB | reference | ✅ |
| [**q8**](https://huggingface.co/mlx-community/LongCat-Video-q8) | 15 GB | 31 GB | 48 GB | very close to bf16 | ✅ |
| [**q4**](https://huggingface.co/mlx-community/LongCat-Video-q4) | 9 GB | 25 GB | 32 GB | minor degradation | ✅ |

All CLIs accept `--variant {auto, bf16, q4, q8}`. Default is `auto` — picks
bf16 if available, falls back to q8 then q4.

## Six task variants, one DiT checkpoint

Each task has a `pipeline_*` module (the library API) and a `run_*.py` CLI under `scripts/`:

| Task | Pipeline module | CLI |
|---|---|---|
| **T2V** — text-to-video | `pipeline_t2v` | `scripts/run_t2v.py` |
| **I2V** — image-to-video | `pipeline_i2v` | `scripts/run_i2v.py` |
| **Video Continuation** | `pipeline_continuation` | `scripts/run_continuation.py` |
| **720p / 30fps refinement** (+ BSA) | `refinement` | `scripts/run_refine.py` |
| **Long-Video** (chained continuation) | `pipeline_long_video` | `scripts/run_long_video.py` |
| **Interactive Video** (per-segment prompts) | `pipeline_interactive` | `scripts/run_interactive.py` |

All six are driven by the same 13.6B DiT with optional LoRAs:
- `cfg_step_lora` collapses CFG branches + reduces sampler step count (`--cfg-step-lora`)
- `refinement_lora` enables the 720p / 30fps refinement pass

Example:

```bash
python scripts/run_t2v.py \
    --weights /path/to/mlx-weights --variant auto \
    --prompt "a fox running through autumn leaves" \
    --num-frames 24 --height 480 --width 832 --seed 42 --out out.mp4
```

## Block Sparse Attention — Metal kernel ladder

The 720p refinement pass uses Block Sparse Attention (sparsity=0.9375,
block_size=64) — each Q block attends to only 6.25% of the KV blocks. Enable it on
a DiT via `dit.enable_bsa(backend=...)`:

| `backend=` | Implementation |
|---|---|
| `"tier_a"` (default) | Pure-MLX reference (correctness fallback) |
| `"metal"` | Auto-selecting: Phase 4 (`simdgroup_matrix` HW matmul) when fp16 + BS=64 + D%32==0 + S≥1280; else Phase 3 (threadgroup-shared K+V); else Phase 2 (simdgroup-cooperative) |
| `"metal_v2"` / `"metal_v3"` / `"metal_v4"` | Explicit Phase 2 / 3 / 4 |

The Tier A backend lives in `models/block_sparse_attention.py`; the Phase 1–4 Metal
kernels live in `models/block_sparse_attention_metal.py`. The refinement CLI selects the
backend via the `LONGCAT_BSA_BACKEND` environment variable (default `tier_a`):

```bash
LONGCAT_BSA_BACKEND=metal python scripts/run_refine.py --weights … --stage1 … --prompt …
```

## Dependencies / reuse

The shared Wan-family modules (Wan VAE, umT5-XXL, base DiT, attention, 3D RoPE, LoRA
loader) are **copy-vendored** into `longcat_video/models/` so the package develops and
releases independently of the Avatar port. The default Flow-Matching scheduler is
provided at runtime by **[`mlx-arsenal`](https://pypi.org/project/mlx-arsenal/)**
(`mlx_arsenal.diffusion.FlowMatchEulerDiscreteScheduler`), a hard runtime dependency
declared in `pyproject.toml` (`mlx-arsenal>=0.10`).

## Install

```bash
pip install -e .                 # runtime: mlx, mlx-arsenal, safetensors, hf_hub, numpy, Pillow, imageio
pip install -e ".[parity]"       # + torch/transformers/diffusers/einops for PT parity tests
pip install -e ".[dev]"          # parity + pytest + ruff
```

## Original plan — all stages shipped

| Stage | Work | Status |
|---|---|---|
| **B0** | Scaffold + copy-vendor reusable modules | ✅ |
| **B1** | Base DiT + LoRAs, T2V pipeline + CLI, publish bf16 | ✅ |
| **B1.6** | LoRA merge wiring across CLIs + cfg_collapse fix | ✅ |
| **B2** | I2V + Continuation pipelines | ✅ |
| **B3** | Coarse-to-fine refinement + BSA (Tier A) | ✅ |
| **B4** | BSA Tier B Metal kernel (Phases 1–4) | ✅ |
| **B5** | Long-Video + Interactive + MOS parity + q4/q8 publish | ✅ |

Per-stage development notes in [`docs/development/`](docs/development/).

## License

MIT. Matches upstream Meituan LongCat-Video and the Avatar companion port.
