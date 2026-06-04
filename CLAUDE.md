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
| Block Sparse Attention (Tier A pure-MLX + Tier B Metal kernel Phases 1–4) | NEW |
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
making structural changes.

### Where lessons live

- **[docs/development/skill-lessons.md](docs/development/skill-lessons.md)**
  — **local lessons (L23+) from this port.** Currently 16 entries
  covering BSA discovery, SDEdit refinement, HF v1.17 CLI changes,
  orchestration sharing invariants, etc.
- [Avatar repo's skill-lessons.md](https://github.com/xocialize/longcat-avatar-mlx/blob/main/docs/development/skill-lessons.md)
  — 22 prior lessons (L1–L22) from the Avatar port. All still apply
  here; we copy-vendored most of those reusable modules.

When both ports are distilled into the `/mlx-porting` skill, L1–L22 +
L23+ merge cleanly into one numbered sequence.

### Capturing new lessons as we go (read this every session)

When you hit a non-obvious gotcha, discover a useful pattern, or save
yourself an hour with a trick, **add an `## L{N}` entry to
`docs/development/skill-lessons.md` in the same shift you discovered
it**. The working memory of *why* fades fast — by the next session
you'll have the symptom but not the diagnosis.

A good lesson has:
1. **Title** — the rule, not the symptom. ("Read published config.json
   before source-spelunking" not "BSA params were hard to find")
2. **What we hit** — failure mode + file:line citation if code-shaped
3. **Rule for next time** — one imperative sentence
4. **Skill update target** — which `/mlx-porting` doc this folds into
   (`common-pitfalls.md`, `numerics.md`, `weight-conversion.md`,
   `publish.md`, etc.)
5. **`[toolkit candidate]` tag** — only when the artifact (helper,
   template, script) is generic enough to extract for reuse across
   multiple ports

Mention the new lesson in your commit message:
`+ L39 (whatever the new lesson is)`. That way `git log` is searchable
for skill updates even without grepping the file.

The lessons file ends with a **running toolkit-candidates tally** —
update it when you tag a new candidate. Extraction trigger is 3+
ports using the same pattern.

## Plan documents

- **[LongCat-Video-Base-MLX-Port-Plan-v1.md](../XDocs/LongCat-Video-Base-MLX-Port-Plan-v1.md)** —
  the 5-week port plan with Tier A → Tier B BSA progression.
