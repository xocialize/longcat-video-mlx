# CLAUDE.md — longcat-video-mlx

Operational orientation for Claude when working in this repo.

## What this is

Apple MLX port of [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's 13.6 B-parameter base text-to-video diffusion model. Six task
variants share a single DiT checkpoint with optional LoRAs:

- **T2V** (text-to-video) → `pipeline_t2v`
- **I2V** (image-to-video) → `pipeline_i2v`
- **Video Continuation** → `pipeline_continuation`
- **Long-Video** → `pipeline_long_video` (chained Continuation)
- **Interactive Video** → `pipeline_interactive` (per-segment prompts)
- (Streamlit UI is out of scope for v1.)

Two LoRAs:
- `cfg_step_lora`: collapses CFG branches + reduces sampler steps
- `refinement_lora`: enables the 720p / 30fps refinement pass (coarse-to-fine)

## Companion repo

[xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
ports the Avatar 1.5 variant of the same architecture (adds audio cross-attn
+ Reference Skip + AudioProjModel). **80 % of this repo's reusable modules
were copy-vendored from there:** Wan VAE, umT5-XXL, base DiT, attention,
blocks, RoPE 3D, LoRA loader. Avatar bug fixes do **not** auto-flow here;
mirror them manually if they touch shared modules.

## What's NEW for the base port (vs. Avatar)

| Component | Status |
|---|---|
| `cfg_step_lora` + `refinement_lora` registration | NEW |
| T2V / I2V / Continuation pipelines | NEW |
| Coarse-to-fine refinement (latent upsampling between passes, LoRA hot-swap) | NEW |
| Block Sparse Attention (Tier A pure-MLX → Tier B Metal kernel) | NEW |
| Long-Video / Interactive orchestration | NEW |

What we **dropped** from Avatar: Whisper encoder, AudioProjModel,
audio_cross_attn / audio_adaLN in DiT blocks, Reference Skip Attention,
3-pass Disentangled CFG combiner. The base port uses standard text-only CFG
(now collapsed via `cfg_step_lora` once loaded).

## Where things live

- **[README.md](README.md)** — public-facing quick start, variants, perf numbers.
- **[longcat_video/](longcat_video/)** — the MLX package. File names mirror
  Meituan's `longcat_video/` layout 1:1 per the mlx-porting skill's
  isomorphic-structure rule.
- **[recipes/](recipes/)** — weight conversion recipe (PT → MLX, with both
  variants and the silent-zero-trap defense).
- **[scripts/](scripts/)** — user-facing CLIs: `run_t2v.py`, `run_i2v.py`,
  `run_continuation.py`, `run_long_video.py`, `run_interactive.py`.
- **[tests/](tests/)** — smoke (no deps, no weights) and parity (PT comparison,
  needs `[parity]` extras and weights via opt-in env vars).
- **[docs/development/](docs/development/)** — Stage B0 recon notes, port plan,
  skill-lessons captured during the port.
- **[docs/model-cards/](docs/model-cards/)** — reference copies of the
  mlx-community model cards (published to HF separately).
- **`refs/`** — reference checkout of `meituan-longcat/LongCat-Video`.
  Gitignored. Re-clone via:
  ```bash
  git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git refs/longcat-video
  ```

## Running tests

```bash
.venv/bin/python -m pytest tests/smoke -v          # always green, no weights
LONGCAT_VAE_AUTO_DOWNLOAD=1 \
  .venv/bin/python -m pytest tests/parity -v        # PT parity, needs [parity] extras
```

Each per-component parity test has its own `LONGCAT_{NAME}_AUTO_DOWNLOAD=1`
env var so you can opt in component-by-component instead of pulling all
~80 GB of source weights at once.

## Skill in use

`/mlx-porting` is the operational skill for this port. Load it before
making structural changes. Per-lesson rationale lives in
[the companion repo's docs/development/skill-lessons.md](https://github.com/xocialize/longcat-avatar-mlx/blob/main/docs/development/skill-lessons.md)
— 22 lessons captured during the Avatar port; all still apply here.

## Plan documents

- **[LongCat-Video-Base-MLX-Port-Plan-v1.md](../XDocs/LongCat-Video-Base-MLX-Port-Plan-v1.md)** —
  the 5-week port plan with Tier A → Tier B BSA progression.
