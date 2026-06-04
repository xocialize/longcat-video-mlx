# longcat-video-mlx

Apple MLX port of [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's 13.6 B-parameter video diffusion model — for inference on Apple
Silicon (M-series).

> **Status: production-tier.** Three variants live on HF — pick by RAM
> budget. Part of the
> [LongCat-Video — MLX](https://huggingface.co/collections/mlx-community/longcat-video-mlx-6a216a3576c098e83c1cc167) collection.
> All six task variants ship and verify; DiT block parity vs PT
> reference is **8.94e-08 (fp32 precision noise)**; the 720p refinement
> pass has a 4-phase Metal kernel ladder culminating in a
> **`simdgroup_matrix` HW-accelerated implementation 2.55× faster than
> dense `mx.fast.scaled_dot_product_attention`** at production sequence
> lengths.
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

CLIs accept `--variant {auto, bf16, q4, q8}`. Default is `auto` — picks
bf16 if available, falls back to q8 then q4. Just download whichever
variant fits your Mac.

## Six task variants, one DiT checkpoint

| Variant | Pipeline | Status |
|---|---|---|
| **T2V** — text-to-video | `pipeline_t2v` | ✅ shipped (B1.3 + B1.4 + golden smoke) |
| **I2V** — image-to-video | `pipeline_i2v` | ✅ shipped (B2.1) |
| **Video Continuation** | `pipeline_continuation` | ✅ shipped (B2.2) |
| **720p / 30fps refinement** | `refinement.py` (+ BSA) | ✅ shipped (B3.1 + B3.2 Tier A + B4.1 Tier B Phase 1–4) |
| **Long-Video** (chained continuation) | `pipeline_long_video` | ✅ shipped (B5.1) |
| **Interactive Video** (per-segment prompts) | `pipeline_interactive` | ✅ shipped (B5.2) |
| Streamlit UI | — | Out of scope (use CLI) |

All six are driven by the same 13.6B DiT with optional LoRAs:
- `cfg_step_lora` collapses CFG branches + reduces sampler step count
- `refinement_lora` enables the 720p / 30fps refinement pass

## Block Sparse Attention — four-phase Metal kernel ladder

The 720p refinement pass uses Block Sparse Attention (sparsity=0.9375,
block_size=64) — a custom routing-then-attend op where each Q block
attends to only 6.25% of the KV blocks. We shipped **five backends**;
the default `--variant`-aware dispatcher picks the right one based on
sequence length and dtype:

| Backend | Speed @ S=12.8K, D=128 | Use case |
|---|---|---|
| **Tier A pure-MLX** | 64 ms (~2× slower than dense) | Reference / correctness fallback |
| **Tier B Phase 1** (naive kernel) | 154 ms | Educational; doesn't beat dense |
| **Tier B Phase 2** (simdgroup-cooperative) | 63.5 ms | fp32 paths, small head dims |
| **Tier B Phase 3** (threadgroup-shared K+V) | 41.1 ms | Any non-fp16 BSA case |
| **Tier B Phase 4** (`simdgroup_matrix` HW matmul) | **25.6 ms** | fp16 + D%32==0 + S≥1280 → **2.55× faster than dense `mx.fast.scaled_dot_product_attention`** |

The auto-selecting dispatcher (`enable_bsa(backend="metal")`) picks
**Phase 4 > Phase 3 > Phase 2** based on constraints + S. Each phase is
a separate skill-lessons entry (L51–L59) documenting the Metal kernel
patterns that worked at each level.

## Reuse from the Avatar port

The Avatar port ([xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx))
already ships the shared architectural components — Wan VAE, umT5-XXL, base
DiT, attention primitives, blocks, 3D RoPE, LoRA loader. This repo
**copy-vendors** those modules (no cross-repo dependency at runtime) so it
can be developed and released independently.

## Original 5-week plan — all stages shipped

| Stage | Work | Status |
|---|---|---|
| **B0** | Scaffold + copy-vendor reusable modules | ✅ |
| **B1** | Convert base DiT + LoRAs, T2V pipeline + CLI, publish bf16 | ✅ |
| **B1.6** | LoRA merge wiring across CLIs + cfg_collapse fix | ✅ |
| **B2** | I2V + Continuation pipelines | ✅ |
| **B3** | Coarse-to-fine refinement + Block Sparse Attention (Tier A) | ✅ |
| **B4** | Block Sparse Attention (Tier B Metal kernel, 4 phases) | ✅ |
| **B5** | Long-Video + Interactive + MOS parity + q4/q8 publish | ✅ |

Full plan in [`../XDocs/LongCat-Video-Base-MLX-Port-Plan-v1.md`](../XDocs/LongCat-Video-Base-MLX-Port-Plan-v1.md).
Per-stage development notes + skill lessons in
[`docs/development/`](docs/development/).

## License

MIT. Matches upstream Meituan LongCat-Video and the Avatar companion port.
