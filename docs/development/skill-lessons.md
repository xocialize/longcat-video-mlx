# Skill lessons — longcat-video-mlx

Continuation of [the Avatar port's skill-lessons](https://github.com/xocialize/longcat-avatar-mlx/blob/main/docs/development/skill-lessons.md).
Avatar lessons are L1–L22; **this port adds L23+**. When both ports are
distilled into the `mlx-porting` skill, the numbering merges cleanly.

**`[toolkit candidate]` tag convention:** same as the Avatar repo — marks
lessons whose artifact (script, helper, template) is a strong candidate for
extraction into a shared `mlx-port-toolkit` repo once the pattern has
survived multiple ports. Confirm by surviving at least 3 of the 4
architecture families (VAE / text encoder / DiT / audio encoder) before
extracting.

Each entry: short title → what we hit → the rule for next time → skill
update target.

## How to add a new lesson (read me before appending)

When you discover a non-obvious pattern, gotcha, or technique during this
port, **add a new `## L{N}` entry here in the same shift you discovered
it**. Don't wait for "end of port" — the working memory of *why* fades
fast.

Checklist for a good lesson entry:

1. **Title** is one line. State the rule, not the symptom. ✅ "Publishing
   config often contains BSA params — read config before source spelunking"
   ❌ "BSA was hard to find"
2. **What we hit** — the failure / symptom / specific moment. File:line
   citation when the lesson is code-shaped.
3. **Rule for next time** — what a *future port should DO*, in one
   imperative sentence.
4. **Skill update target** — which doc in `/mlx-porting` skill this folds
   into (`common-pitfalls.md`, `repo-layout.md`, `numerics.md`,
   `weight-conversion.md`, `publish.md`, etc.). If it's a new section,
   say so.
5. **`[toolkit candidate]` tag** — only when the artifact is generic
   enough to recur across multiple ports.

After adding the lesson, mention it in your commit message
(e.g. `+ L27 (BSA Tier A correctness gate via sparsity=0)`).

---

## L23. Publishing config often contains the implementation details — read it before source-spelunking `[toolkit candidate]`

**What we hit:** Plan v1 flagged the **BSA block configuration** as the #1
blocker for Tier B (Metal kernel) — chunk size, sparsity, and top-k all
depend on it. We expected to spend an afternoon source-spelunking
`block_sparse_attention/bsa_interface.py` to back out the constants. Then
we opened `dit/config.json` from the published model and:

```json
"bsa_params": {
  "sparsity": 0.9375,
  "chunk_3d_shape_q": [4, 4, 4],
  "chunk_3d_shape_k": [4, 4, 4]
},
"enable_bsa": false,
```

All three constants were sitting right there. Plan v1 Open Question #1
was closed by reading one file.

**Rule for next port:** Before opening any modeling source file to back
out architectural constants, **diff the published config.json against
the model's `__init__` signature**. Anything in the config that isn't a
kwarg of `__init__` is a *runtime-mutable* parameter the upstream team
chose to expose — and these are exactly the constants downstream code
needs to know about (block shapes, sparsity, dropout schedules, head-dim
splits, etc).

This shows up at the top of the workflow doc as a Stage-0 task:
> Snapshot all published `config.json` files. Cross-reference any
> "magic numbers" the plan flagged as Open Questions against them
> *before* opening modeling source.

**Skill update target:** new entry in `repo-layout.md` under "Stage 0
reconnaissance — what to do before code." Tagged toolkit-candidate
because the diff itself (config-vs-init) is mechanical and reusable.

**Artifact location:** `docs/development/bsa-config-found.md`.

---

## L24. BSA / sparse-attention Tier A correctness gate: `sparsity=0 ≡ dense SDPA` `[toolkit candidate]`

**What we hit:** Block Sparse Attention is a routing-then-attend op:
mean-pool blocks → top-k routing → mask out non-routed pairs → attend.
Many places to make off-by-one errors (block index, mask polarity,
softmax scale, etc.). Wanted a single end-to-end correctness gate before
spending hours on per-step parity vs PT.

The strongest possible gate is the **degenerate case where routing keeps
every block** — i.e. `sparsity=0`. In that regime BSA must produce
**exactly the same output as dense scaled-dot-product attention**:

```python
out_bsa   = bsa_attention(q, k, v, shape=(T,H,W), sparsity=0.0)
out_dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=...)
assert mx.max(mx.abs(out_bsa - out_dense)) < 1e-4
```

If that holds, **every step of the routing + masking pipeline is
verified** — because any error (wrong block index, flipped mask polarity,
missing scale) would diverge on this case. Took the BSA test suite from
"I think it works" to "verified" in 4 lines.

The drift tolerance is ~1e-4 (not 0) because the additive-0 mask path
still adds a `0.0` tensor before softmax, which has different rounding
behavior than the no-mask path.

**Rule for next port:** For any sparse / routed / gated attention port,
write the degenerate-case test first:
- Sparse attention with sparsity 0 (or top-k = all blocks) ≡ dense
- MoE with all experts → expert 0 ≡ that expert's dense forward
- Linear attention with kernel that reduces to softmax ≡ standard attn

Land that test in smoke before any routing tests. It's the cheapest
"my whole pipeline is wrong" alarm you'll ever build.

**Skill update target:** new section in `numerics.md` titled "Degenerate-
case correctness gates for sparse/routed ops." Tagged toolkit-candidate
because the pattern (find a degenerate setting that reduces to a known-good
dense reference) is reusable across MoE, BSA, low-rank, etc.

**Artifact:** `tests/smoke/test_bsa_tier_a_smoke.py::test_bsa_sparsity_zero_matches_dense`.

---

## L25. Conversion exits 0 but missing artifacts → check ALL expected files

**What we hit:** The base DiT + LoRA conversion script exited 0 and the
DiT was on disk (6 shards, 26 GB) — but the LoRA directory was
**completely absent**. No error log (background tee was disconnected).
The recipe quietly skipped the LoRA step.

A second run of the *same* recipe finished the LoRAs in ~3 minutes
because the skip-done sentinels short-circuited the already-finished
components. The bug must have been in the first run's mid-step exit
handling, but we never had to find it — the resumable recipe pattern
made it free to recover from.

**Rule for next port:**

1. After any "done" notification on a conversion, **`ls -R` the output
   dir against the expected layout** — exit code 0 + main outputs present
   is NOT proof of completion.
2. Build conversion recipes with **skip-done sentinels per component**
   (`if (out / "component_name" / "marker.safetensors").exists(): skip`).
   Idempotent re-run cost is one process startup + one filesystem walk;
   non-idempotent re-run cost is hours of re-conversion. Always take
   the cheap option.

The Avatar port's recipe already follows this; the lesson here is to
make the **post-conversion validation step explicit** in the workflow,
not implicit in "well the exit code was 0."

**Skill update target:** new entry in `weight-conversion.md` titled
"Validate output layout after every conversion run." Cross-references the
Avatar L4 (toolkit candidate: Conv*d transpose / gamma-skip) — same
defensive posture.

---

## L26. Empty tee output from background shells: `&` + shell exit drops the redirect

**What we hit:** We launched a long-running conversion as
`.venv/bin/python ... 2>&1 | tee /tmp/log &`. The parent shell exited
right after starting the background job. The `tee` process inherited the
shell's stdio, which got disconnected → log file ended up zero bytes
even though the Python process ran for ~25 minutes and produced output.

Same pattern bit us twice — once for conversion, once for publish.

**Rule for next port:**

- Use `nohup cmd > log 2>&1 &` for true detached background runs
  (writes directly to the file without a tee in the middle)
- OR: foreground the process and let the harness's background-task tool
  capture stdout (harness keeps the pipe alive even when your turn ends)
- Never trust `cmd 2>&1 | tee log &` from a script — the lifecycle of
  `tee` is tied to the launcher's stdio

Best workaround we found: when the log is empty but you have other
ground-truth signals (file appearances, `pgrep`), trust those instead and
move on. Don't try to recover the missing log.

**Skill update target:** new entry in `common-pitfalls.md` under
"Long-running ops and background processes."

---

## L27. Implicit sibling-venv deps catch on a fresh venv install

**What we hit:** `pipeline_t2v.py` imports `from mlx_arsenal.diffusion
import FlowMatchEulerDiscreteScheduler` and `scripts/run_t2v.py` imports
`from transformers import T5TokenizerFast`. Neither was listed in
`pyproject.toml` — both happened to be installed in the Avatar repo's
.venv, which our development venv had inherited some packages from. On
the first publish/golden run from a clean state, both failed.

**Rule for next port:** Run a `freshness` check before publishing:

```bash
# Bash: install only what pyproject declares + run smoke + run CLI --help
python -m venv /tmp/fresh && \
  /tmp/fresh/bin/pip install -e . && \
  /tmp/fresh/bin/python -m pytest tests/smoke && \
  /tmp/fresh/bin/python scripts/run_t2v.py --help
```

If the fresh-venv smoke passes but CLIs fail with `ModuleNotFoundError`,
the deps are wrong. Repo-local venv passes the test because of
cross-repo inheritance.

For this port: added `mlx-arsenal>=0.10` to runtime deps; `transformers`
stays in `[parity]` (the CLI now explains the install command when it
hits the ImportError).

**Skill update target:** new entry in `repo-layout.md` under a "Release
sanity checks" subsection. Tagged toolkit-candidate if we can make the
fresh-venv check a reusable script.

---

## L28. HF CLI v1.17 command surface (post-`huggingface-cli` rename)

**What we hit:** `huggingface-cli` was renamed to `hf` in 2025. Several
of Avatar's L19 commands changed shape:

| Avatar (huggingface-cli) | Current (`hf` 1.17) |
|---|---|
| `huggingface-cli whoami` | `hf auth whoami` |
| `huggingface-cli repo create --exist-ok` | `hf repos create --exist-ok` (note: **plural** — `hf repo` still works but is deprecated) |
| `huggingface-cli upload` | `hf upload REPO_ID LOCAL_PATH PATH_IN_REPO` |
| `huggingface-cli api /api/models/X` | `hf models info X` (no general `api` command) |

Collections still work via `hf collections create/add-item/list`.
**Collection description has a 150-char limit** — split the longer
description into the model card README + a short collection blurb.

**Rule for next port:** Pin `huggingface_hub>=0.34` in pyproject and use
the `hf` commands above. The Avatar port's L19 still encodes the right
*shape* (create-then-upload, exist-ok idempotency) but the literal
commands have shifted.

**Skill update target:** update `publish.md` with the new commands.
Avatar L19 stays as the canonical pattern; add a v1.17 command table.

---

## L29. Stage README.md from `docs/model-cards/` before uploading

**What we hit:** HF picks up `README.md` at the repo root as the canonical
model card page. We keep the source-of-truth in
`docs/model-cards/bf16.md` so it's git-tracked alongside code, but the
upload target needs `README.md`. Easy footgun: forget to stage and the
HF page renders an auto-generated placeholder.

**Rule for next port:**

```python
# In the publish script:
def stage_readme(variant_dir, model_card_md):
    shutil.copy2(str(model_card_md), str(variant_dir / "README.md"))
```

Run the stage step **before** `hf upload`. If you're using `hf upload`'s
`--exclude` filter, exclude the source `docs/model-cards/` dir but
include the staged `README.md` at the variant root.

**Skill update target:** add to `publish.md` as a required step in the
publish flow.

---

## L30. Statistical signature of a tiny golden distinguishes real generation from noise

**What we hit:** Wanted to verify the bf16 T2V pipeline works
end-to-end without spending 30 minutes on a 50-step run. A 4-step run is
visually garbage but should still produce *real diffusion trajectory
statistics*, not pure noise.

Pure-noise output signature:
- mean ≈ 127 (centered uniform)
- std ≈ 73 (uniform over [0, 255])
- per-channel std ≈ uniform (R, G, B all ~73)
- inter-frame mean abs diff ≈ 85 (no temporal correlation)

Real-generation 4-step output signature (cat surfing prompt):
- mean = 101 (image-like centering)
- std = 63 (structured, not uniform)
- per-channel std: R=58, G=57, B=61 (per-channel structure)
- inter-frame mean abs diff: 67, 23, 32, 19 (first frame is the
  noise→image transition; later frames have temporal continuity)

That diverges enough from the noise baseline to call the pipeline
**verified end-to-end** in seconds, without needing visual inspection or
full quality.

**Rule for next port:** When running a golden under a tight step budget,
inspect:
1. `arr.mean()`, `arr.std()` — should NOT match uniform-noise statistics
2. Per-channel std — should vary across R / G / B
3. Inter-frame difference — should decrease after the first transition

If all three match noise statistics, the model is producing junk
regardless of step count. Use this as the cheap "pipeline alive" check
before scheduling a long golden.

**Skill update target:** new entry in `validation.md` (or `numerics.md`
if no validation doc yet) titled "Tiny-golden statistical signature."

---

## L31. MLX has no native trilinear; do separable bilinear+linear in fp32

**What we hit:** Refinement upsamples a `[1, 3, T_old, H_old, W_old]`
video to `[1, 3, T_new, H_new, W_new]` via trilinear interpolation. MLX
has no native 5-D trilinear. Tried `mx.image` — only 2-D.

The clean alternative: **separable trilinear = temporal linear ∘
spatial bilinear** (the three axes are independent, so the order doesn't
matter and the result is identical to true trilinear). MLX has neither
1-D linear nor 2-D bilinear as a primitive — both are implemented via
`mx.take` + linear interpolation:

```python
ts = mx.linspace(0.0, T - 1, num=new_T).astype(mx.float32)
t0 = mx.floor(ts).astype(mx.int32)
t1 = mx.minimum(t0 + 1, T - 1)
wt = (ts - t0.astype(mx.float32))[None, None, :, None, None]
a = mx.take(x, t0, axis=2); b = mx.take(x, t1, axis=2)
result = a * (1.0 - wt) + b * wt
```

Critically, **do the whole resize in fp32** even when the rest of the
forward is bf16. Resampling in bf16 spuriously washes detail and
contaminates the VAE encode step downstream.

**Rule for next port:** When you need a higher-D resampling primitive
MLX doesn't ship, decompose into per-axis 1-D linear (separable) and run
in fp32. Cast back to the model's working dtype after.

**Skill update target:** new entry in `numerics.md` titled "Resampling
in MLX — separable + fp32." Tagged toolkit-candidate if the helper
generalizes to N-D.

---

## L32. SDEdit-style refinement: partial-noise + truncated schedule + frozen cond

**What we hit:** Refinement (`refinement.py`) is NOT fresh sampling from
pure noise. It's SDEdit-style: VAE-encode the upsampled coarse video,
inject partial noise at threshold τ, then denoise only the
`timesteps ≤ τ*1000` portion of the schedule.

Three traps:

1. **Schedule truncation must happen on the scheduler's `timesteps`
   array, not just the loop range.** Insert τ*1000 itself at the head:
   `timesteps = [τ*1000, then timesteps where t < τ*1000]`.
2. **Cond latents are frozen at t=0 throughout** —
   `timestep[:, :num_cond_latents] = 0` AND `scheduler.step` updates
   only the noise slice (`latents[:, :, num_cond_latents:]`).
3. **No CFG** — refinement_lora is trained at guidance_scale=1.0, so the
   loop does a single forward per step. 50 nominal refinement steps ≈
   25 actual denoising steps, ≈ same wall-time as 25 baseline T2V steps
   (each of which is 2 forwards for CFG).

Found these by literally diffing our pipeline against upstream's
`generate_refine` (lines 1098-1340) and missing all three on the first
pass.

**Rule for next port:** When porting any "refinement" / "img2img" /
"high-res fix" pipeline, treat it as a *separate flow* from the main
sampling pipeline. Specifically check:

- Are timesteps truncated? (look for `t_thresh`, `cutoff`, `start_step`)
- Are some latents frozen at t=0? (look for `num_cond_latents`,
  `timestep[:N] = 0`)
- Is CFG disabled? (look for `guidance_scale=1.0` defaults in the run
  script, and whether the loop does 1 or 2 DiT forwards)

Don't copy your baseline sampler's structure — refinement is a different
algorithm.

**Skill update target:** new section in a new doc `refinement-passes.md`
under `mlx-porting/concepts/`. Covers SDEdit, ControlNet-style guidance,
high-res fix — all variations of the same partial-noise-denoise pattern.

---

## L33. Orchestration pipelines must share sub-pipeline component instances

**What we hit:** Long-Video and Interactive pipelines internally build a
T2V pipeline + a Continuation pipeline. Naive impl would let each
sub-pipeline load its own VAE / umT5 / DiT — but those are 26 GB / 11 GB
/ 242 MB. Two copies of the DiT alone would blow unified memory on a
64 GB machine.

The orchestrator must construct the sub-pipelines with the **same component
instances**:

```python
self.t2v = LongCatVideoT2VPipeline(vae, text_encoder, dit, config=...)
self.continuation = LongCatVideoContinuationPipeline(vae, text_encoder, dit, config=...)
```

And we lock this with an explicit test:

```python
def test_subpipelines_share_components():
    pipe = Pipeline(vae=vae_stub, ...)
    assert pipe.t2v.vae is pipe.continuation.vae   # `is`, not `==`
    assert pipe.t2v.dit is pipe.continuation.dit
```

**Rule for next port:** When an orchestration layer holds two or more
sub-pipelines that share model components, write a `is`-identity test
in smoke. Catches the "I'll just re-instantiate inside" refactor before
it lands.

**Skill update target:** add to `common-pitfalls.md` under a new
"Orchestration / multi-stage pipelines" section.

---

## L34. Mixed-Q-and-K-seq attention should opt out of BSA, not toggle per-block

**What we hit:** BSA assumes a uniform 3-D token grid where Q and K
share the same `(T, H, W)` layout. But several paths in the DiT have
mixed Q/K sequence lengths:

- KV-cache continuation: Q is the new chunk, K includes the cached prefix
- Cond-noise branch: Q = noise tokens, K = full cond + noise

If you try to run BSA on these, the block indexing breaks (Q-block ids
don't map to K-block ids cleanly).

Cleanest fix: have `_process_attn(q, k, v, shape=None)` accept a 3-D
`shape` hint and **fall back to dense SDPA when `shape is None`**. The
caller passes `shape=None` for any branch where Q-seq ≠ K-seq:

```python
if self.enable_bsa and shape is not None and q.shape[-2] == k.shape[-2]:
    return bsa_attention(q, k, v, shape=shape, ...)
return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
```

No per-block toggles, no special-case branches. The `shape` parameter
naturally encodes "this is a uniform-grid self-attention" — anything else
is dense.

**Rule for next port:** When wiring a structured-attention variant (BSA,
windowed, axial), make the structured path *opt-in via the shape hint*
and dense the default. Don't try to make the structured op handle all
the corner cases — let the call site decide.

**Skill update target:** add to `common-pitfalls.md` under
"Sparse/structured attention call sites."

---

## L35. Document-level milestones belong in MEMORY, not commit messages

**What we hit:** Across this port we landed 10 PRs in one day. Each PR
has a useful body but the *cross-PR narrative* ("we discovered BSA
params in the config, which closed Open Question #1 from the plan; then
we shipped T2V → I2V → Continuation → Refinement → BSA Tier A in
sequence") only lives in the user's auto-memory.

If the user comes back in 3 weeks asking "what's the state of LongCat?"
the natural answer is "look at the memory checkpoint" — not "scan 10
PR bodies in order." We did this correctly in this port by updating
`MEMORY.md` at the end with a port-level summary entry.

**Rule for next port:** At every meaningful milestone (weights published,
all variants shipped, parity hit), write a one-paragraph milestone entry
into the user's `MEMORY.md` linking to a per-port memory doc with the
PR list, HF artifacts, statistics, and pending work. The doc is the
**resumption point** for the next session.

**Skill update target:** add to `workflow.md` (if it exists) or
`repo-layout.md` under "Session continuity."

---

## L36. `mx.argpartition` for top-k: kth-element-bottom, flip sign for top

**What we hit:** BSA top-k routing needs the indices of the K largest
scores. MLX's `mx.argpartition(x, kth, axis)` returns indices arranged
so the `kth` position holds the kth-*smallest* value — bottom-k by
default. For top-k, negate:

```python
# Top-k by score value:
neg_score = -score
part = mx.argpartition(neg_score, kth=num_selected - 1, axis=-1)
top_k_indices = part[..., :num_selected]
```

The `kth=num_selected - 1` (not `num_selected`) is also a footgun — MLX
matches NumPy's convention where `kth` is the *index* of the partition
point, not the *count* of items below it.

**Rule for next port:** For top-k in MLX:
- `argpartition(x, k-1)` returns indices where positions `[0:k]` are
  the k smallest (in some order)
- For top-k by VALUE, negate the input first
- Sanity-check by computing `take_along_axis(x, top_k, ...)` and
  verifying all selected values are above the unselected median

**Skill update target:** new entry in `mlx-builtins.md` (if exists) or
`numerics.md` titled "Top-k via `mx.argpartition`."

---

## L37. Token-block index map: pure-Python flat-index → 3-D block coord lookup

**What we hit:** BSA needs to expand a per-Q-block-pair routing decision
to a per-token-pair attention mask. Naively this would require a giant
gather over the routing matrix. Cleaner approach: precompute a
`[S]` lookup table `token_block_index[s]` that gives the block id for
each flat token index `s`:

```python
s = mx.arange(T * H_ * W, dtype=mx.int32)
t = s // (H_ * W)
hw = s % (H_ * W)
h = hw // W
w = hw % W
bt = t // cT; bh = h // cH; bw = w // cW
return bt * (Bh * Bw) + bh * Bw + bw
```

Then the token-pair mask is just two `mx.take`s:

```python
expanded_q = mx.take(block_pair_mask, token_block_index, axis=-2)
token_pair_mask = mx.take(expanded_q, token_block_index, axis=-1)
```

Result: `[B, H, S, S]` bool from `[B, H, num_q_blocks, num_kv_blocks]`
bool in two ops. No loops, no scatter, no kronecker.

**Rule for next port:** When expanding a coarse-resolution decision
(block-level routing, patch-level scores, etc.) to a fine-resolution
target (per-token mask, per-pixel attention), precompute the
**coarse-to-fine index table** once and use `take` along each axis.
Cleanest pattern we've found in MLX for this class of op.

**Skill update target:** new section in `numerics.md` titled
"Coarse-to-fine expansion via per-axis `take`." Tagged toolkit-candidate
if the index table generalizes (it does — same pattern works for
windowed attention, axial decompositions, hierarchical routing).

---

## L38. Skip-done sentinels per component make re-runs free `[toolkit candidate]`

**What we hit:** This is a reinforcement of patterns already in the
Avatar port, but bears explicit calling-out because it saved us hours.

Every conversion / build step that writes a multi-GB artifact should
check whether that artifact already exists before re-doing the work:

```python
def _component_done(out_dir, component, marker_file):
    return (out_dir / component / marker_file).exists()

if skip_done and _component_done(out_dir, "dit",
                                 "diffusion_pytorch_model.safetensors.index.json"):
    print("  DiT already converted — skipping")
else:
    convert_dit(out_dir)
```

In our case, the DiT conversion took ~25 min and the umT5 took ~15 min.
When the LoRA step silently failed (L25), the second run of the same
recipe completed both LoRAs in ~3 min because everything else was
already done. **The recipe is its own checkpoint.**

**Rule for next port:** Every multi-step conversion / build recipe must
have **per-component skip-done sentinels keyed off real output files**
(not arbitrary marker files — use the actual artifacts the step
produces). Re-running the recipe should always be safe and ~instant if
nothing changed.

**Skill update target:** strengthen the existing guidance in
`weight-conversion.md` with the per-step pattern shown above. Cross-ref
L25 (post-conversion validation is the safety net; skip-done sentinels
are the recovery mechanism).

---

## L39. LoRA merge wrapper must FAIL LOUDLY when 0 modules match `[toolkit candidate]`

**What we hit:** When wiring `cfg_step_lora` and `refinement_lora` into
the 5 CLIs, we built a thin `merge_lora(dit, variant_dir, name)` wrapper
around the existing `merge_lora_into_model`. The underlying merge already
returns `{"applied": [...], "unmapped": [...]}` lists, but if the LoRA
key encoding was even slightly off (e.g. a future LoRA uses a different
separator), the function would happily return `{"applied": [],
"unmapped": [all_paths]}` and **the inference would silently run with
the base model** — producing wrong output that looks like a slow-quality
issue but is actually a no-op merge.

Avoid this by asserting in the wrapper:

```python
result = merge_lora_into_model(dit, sd, multiplier=multiplier)
n_applied = len(result["applied"])
assert n_applied > 0, (
    f"{name}: 0 modules merged — LoRA target paths don't match the "
    f"DiT module tree. A no-op merge would silently produce wrong "
    f"output, so failing loudly. Check lora.decode_module_name's "
    f"output against dict(tree_flatten(dit.parameters())).keys()."
)
```

**Rule for next port:** Any "merge weight delta into model" helper —
LoRA, LyCORIS, low-rank adapters, peft-style overlays — must fail loudly
when the merge is a no-op. The wrapper layer (CLI / orchestration) is
the right place to enforce it: the underlying merge math has no
business deciding whether 0 applications is an error. Treat the
**absence of effect** as a bug, not as silence.

This is L24's principle (degenerate-case correctness gate) applied to
the *operational* layer: instead of testing the algorithm reduces to
something known-good, test that the *side effect* the caller asked for
actually happened.

**Skill update target:** add to `common-pitfalls.md` under
"Silent side-effect failures" — same family as the `mx.eval` before
`mx.save_safetensors` defense (zero-tensor save trap from L11 in Avatar
context). Tagged toolkit-candidate because the pattern (assert>0 in
the wrapper, not the math) generalizes to every "patch into module
tree" helper across MLX ports.

**Artifact:** `scripts/_common.py::merge_lora` +
`tests/smoke/test_lora_merge_wrapper_smoke.py`.

---

## L40. Free the LoRA state dict immediately after merge

**What we hit:** Pre-merging `cfg_step_lora` (2.3 GB) and
`refinement_lora` (3.0 GB) into the DiT — the LoRA state dict is loaded
as a Python dict of `mx.array`s totaling ~3 GB. After
`merge_lora_into_model` has finished, those tensors are *redundant*: the
delta is already in the DiT weights. But the state dict stays
referenced in the calling scope until garbage collection runs.

For inference at 480p the spare 3 GB is fine; for refinement at 720p
with BSA on (which materializes the full S² mask per layer × 48 layers),
unified memory is *much* tighter and a held 3 GB LoRA state dict can
push us over.

Drop it explicitly:

```python
result = merge_lora_into_model(dit, sd, multiplier=multiplier)
...
del sd                # release ~2-3 GB before returning
return result
```

For the wrapper to do this, the loader has to be a separate call (so
the wrapper owns the only reference). Hence the
`load_lora_state_dict(path) → sd` helper as a distinct step before
`merge_lora_into_model(dit, sd)`.

**Rule for next port:** Whenever you load weights specifically to
*compute a delta then discard*, the loader and the consumer should be
separate calls in the same scope, and the consumer's scope should
`del` the source immediately. MLX's lazy eval + unified memory means
references held in Python keep tensors resident even after the model
has "absorbed" them.

**Skill update target:** add to `numerics.md` under a new section
"Unified memory hygiene during weight ops." Cross-references the
silent-zero-trap (`mx.eval` before save) — same kind of
"physical-memory-vs-logical-state" mismatch.

---

## L41. Identical fast-mode flip across all CLIs → the orchestration layer is where it belongs

**What we hit:** The `cfg_step_lora` merge is followed *every time* by
the same 4 lines:

```python
cfg.cfg_collapse = True
cfg.num_sampling_steps = 8
cfg.text_guidance_scale = 0.0
print(...)
```

Across 5 CLIs (`run_t2v`, `run_i2v`, `run_continuation`,
`run_long_video`, `run_interactive`). The orchestration variants
(Long-Video, Interactive) additionally need to flip the same on BOTH
sub-pipelines (`pipeline.t2v.config` + `pipeline.continuation.config`).

Today this is duplicated. The right shape is probably:

```python
# In _common.py:
def apply_cfg_step_lora(pipeline, dit, variant_dir):
    merge_lora(dit, variant_dir, "cfg_step_lora")
    pipeline.set_fast_mode(num_steps=8)   # pipeline knows its own sub-configs
```

We didn't do this yet because the 4-line duplication is small and the
shape of `set_fast_mode` would need to handle Long-Video / Interactive's
two-sub-config case specifically. **Decision: leave the duplication for
now; revisit when a 6th CLI shows up or a config field gets added.**

**Rule for next port:** When wiring an opt-in capability (LoRA, BSA,
quantization) across multiple CLIs, the moment you write the *third*
near-identical block, move the orchestration step into the pipeline
class or a shared `_common.py` helper. Two copies is a coincidence;
three is a pattern.

**Skill update target:** add to `repo-layout.md` under
"CLI-vs-pipeline boundary" — the heuristic for "is this CLI plumbing
or pipeline behavior?"

---

## L42. Two-pass and single-pass CFG branches must share timestep-shape normalization

**What we hit:** The 2-pass baseline CFG path (`cfg_collapse=False`)
had a defensive `if timestep.ndim == 0: timestep = timestep[None]` line
right before `mx.repeat`. The single-pass `cfg_collapse=True` branch
didn't — because it doesn't repeat. So a scalar (0-d) timestep from
the scheduler trickled straight through to the DiT, which only
checks `ndim == 1` for its `[B] → [B, N_t]` broadcast.

Downstream consequence: `timestep.flatten()` on a 0-d array gives shape
`(1,)`, not `(B*N_t,)`. The t_embedder produces a `[1, 512]` tensor;
the `.reshape(B=1, N_t=2, -1)` call then *fits* that into shape
`(1, 2, 256)` — silently corrupting the per-frame embedding dim from
512 to 256.

The crash showed up 200ms later at the next-block's `adaLN_modulation`
linear, which expects 512-dim input. The error message **didn't point
at the timestep** — it pointed at `addmm` shape mismatch in the
modulation layer. Took source-walking the DiT forward to backtrack to
the timestep flatten.

This had been latent in the codebase since B1.3 (T2V pipeline). The
baseline smoke test (4 steps × 5 frames, `cfg_collapse=False`) couldn't
trigger it because the 2-pass branch had the normalization. The bug
only manifested when we wired LoRA merge + flipped `cfg_collapse=True`
in B1.6.

**Rule for next port:** Any branch that takes a "scalar OR 1-D" input
must do shape normalization in EVERY branch, not just the one that has
a downstream op that errors loudly. The 2-pass branch had `mx.repeat`
which would have errored on a 0-d input — that's WHY the normalization
was there. The single-pass branch had no such loud-erroring op, so the
normalization was forgotten — and the silent corruption rode all the
way to the next-block.

**Belt-and-suspenders fix:** normalize at the DiT entry as well:

```python
# Normalize scalar (ndim==0) → [B=1], then expand [B] → [B, N_t]
if timestep.ndim == 0:
    timestep = timestep[None]
if timestep.ndim == 1:
    timestep = mx.broadcast_to(timestep[:, None], (B, N_t))
```

That way no future call site can trip on the same trap — even if a
test stub forgets the normalization, the model handles it.

**Regression-test pattern that catches this in smoke:**

```python
class StubDiT:
    def __call__(self, lat, t, *a, **kw):
        received_ndim.append(int(t.ndim))
        return mx.zeros_like(lat)

pipe = LongCatVideoT2VPipeline(..., dit=StubDiT(), config=cfg_collapse_True)
pipe._cfg_forward(latents=..., timestep=mx.array(500.0), ...)
assert received_ndim == [1]
```

Stub the DiT, assert it receives a 1-D timestep. Locks the invariant
without needing weights.

**Skill update target:** add to `common-pitfalls.md` under a new
"Branch parity" section. Cross-references L34 (mixed Q/K seq → opt out
of BSA, don't toggle per-block) — same family: when you have two
branches, audit them for *every* normalization the other branch does.

---

## L43. Quantize-from-already-converted-bf16, don't re-walk the PT source `[toolkit candidate]`

**What we hit:** Naive impl of `build_q_variant` would have called
`convert_dit` again with `quantize_bits=4` — re-downloading the 54 GB
PT source, re-loading sharded fp32 weights, re-casting to bf16,
quantizing, and saving. That's ~30 min for q4 + another 30 min for q8.

Better: quantize **from the already-converted bf16 DiT on disk**. The
DiT module tree is the same, the weight layout is the same, the
parameter names are the same. Just:

```python
model = LongCatVideoTransformer3DModel.from_config(bf16_dit_cfg)
for shard in bf16_shards:
    model.load_weights(str(shard), strict=False)
nn.quantize(model, bits=bits, class_predicate=...)
quantized_sd = dict(tree_flatten(model.parameters()))
save_sharded(quantized_sd, out_dir)
```

Real impact: q4 + q8 both finished in under a minute. Compare to the
original bf16 conversion which took ~25 minutes (had to pull 54 GB from
HF + fp32 → bf16 cast).

**Rule for next port:** When adding quantized variants on top of an
existing bf16 variant, never re-walk the source PT weights. Build the
model with the bf16 config, load the bf16 weights, run `nn.quantize`,
snapshot the resulting parameter tree, save. The recipe-side
`from_bf16` orchestration is short (~20 lines) and idempotent —
re-running it is free (skip-done sentinels).

**Skill update target:** add to `weight-conversion.md` as a section on
"Quantized variants — quantize-from-bf16 pattern." Tagged toolkit-
candidate because the helper (load bf16 → quantize → save sharded)
generalizes to every quant variant we ship across all ports.

**Artifact:** `recipes/convert_longcat_video.py::quantize_dit_from_bf16`.

---

## L44. Same `class_predicate` in both the recipe and the runtime loader `[toolkit candidate]`

**What we hit:** Quantization with selective skips needs a predicate
function used **twice**:

1. **At conversion time:** `nn.quantize(model, class_predicate=predicate)`
   — only the matched Linears get quantized, the skipped ones stay at bf16.
2. **At runtime:** before `dit.load_weights(quant_shards)`, you must call
   `nn.quantize(dit, class_predicate=predicate)` with the SAME predicate
   so that `QuantizedLinear` modules are installed at exactly the matching
   paths — otherwise `load_weights` will try to load 4-bit packed tensors
   into a regular `Linear` and fail (or worse, silently misload).

Both call sites need the same skip_patterns list. We solved this by:
- Embedding `skip_patterns` in the **published config.json** under
  `quantization.skip_patterns`
- The conversion recipe reads its skip list from `DIT_QUANT_SKIP_PATTERNS`
- The runtime loader reads them from `quant_cfg["skip_patterns"]` with
  a fallback to the same default list

This way the variant is **self-describing**: anyone who downloads
`LongCat-Video-q4/` from HF can apply the right predicate just by
reading `dit/config.json`. No version mismatch between recipe and runtime.

**Rule for next port:** Any quant config that uses selective skips
must embed the skip patterns in the variant's config.json. The
predicate function (and its skip list) is part of the model's *contract*
with consumers — not a recipe implementation detail.

**Skill update target:** add to `weight-conversion.md` and to a new
`quantization.md` page covering both conversion + runtime sides. Tagged
toolkit-candidate.

**Artifacts:**
- `recipes/convert_longcat_video.py::_write_dit_config_with_quant`
- `recipes/convert_longcat_video.py::_should_quantize_dit_linear`
- `scripts/_common.py::_apply_quantization_for_load`

---

## L45. Quant variants reuse non-DiT components verbatim `[toolkit candidate]`

**What we hit:** Naive impl would have separately converted VAE, umT5,
LoRAs, scheduler, tokenizer for each quant variant — each variant's
output dir gets its own copy of everything. That's 11 GB umT5 × 3
variants = 33 GB of redundant text encoder shards.

But the VAE, umT5, LoRAs are **not quantized** — they're identical bytes
in all three variants. Quantizing them would degrade output more than
save space (rule of thumb: only quantize components > ~5 GB; below that
the absolute disk savings don't justify the quality drift).

Just **copy from bf16** in the q-variant build:

```python
import shutil
shutil.copytree(bf16_dir / "vae", q_dir / "vae")
shutil.copytree(bf16_dir / "text_encoder", q_dir / "text_encoder")
shutil.copytree(bf16_dir / "lora", q_dir / "lora")
```

(HF stores these as deduplicated content-addressable blobs server-side
anyway, so the upload cost is just the q-variant DiT.)

**Rule for next port:** When shipping multiple quant variants of the
same model, only quantize the components big enough to matter
(typically > 5 GB) and copy the rest verbatim. The smaller components
are sensitive — quantizing them often degrades output quality more
than the marginal disk savings justify.

**Skill update target:** add to `weight-conversion.md` under quant
variants. Tagged toolkit-candidate because the
"DiT-only-quant, copy-everything-else" pattern is the dominant shape
for diffusion model quant variants.

**Artifact:** `recipes/convert_longcat_video.py::build_q_variant`.

---

## L46. HF upload exit isn't atomic with upload completion

**What we hit:** The publish script wraps `subprocess.run(["hf", "upload",
...], check=True)` which blocks until the subprocess returns. When it
returned for q8, the log only showed "Finished hashing 22 files" — no
"Done." final line. Checking HF API immediately showed 1 file
(.gitattributes), 0 bytes — looked like the upload silently failed.

After a delay (probably async commit settling on HF's backend), the same
log file gained a final commit-URL line: `url=https://huggingface.co/.../
commit/87131df...`. The upload had completed, just hadn't flushed the
final lines to stdout yet — Python's subprocess pipe buffering.

So the apparent "publish failure → 1 file on HF" was actually:
- `hf upload` HAD finished
- The commit HAD landed
- The script's final `print("Done.")` had executed
- ... but Python's output buffer hadn't been flushed before the harness
  considered the task complete
- Polling HF API right after the task completion still showed the
  pre-commit state (stale)

After ~30s the HF API caught up and reported the full state.

**Rule for next port:** Don't trust `hf models info` polled
immediately after an upload completes. Either:
- Sleep 30+ seconds before the verification poll
- Or read the publish script's commit URL from the log as the
  ground-truth signal (it's flushed before the script's final print)
- Or just trust the script's exit code and accept that HF API
  consistency takes a few seconds to settle

**Skill update target:** add a paragraph to `publish.md` under
"Verification" — the polling pattern that actually works.

---

## L47. Run upstream PT references via `sys.modules` swap + stubs `[toolkit candidate]`

**What we hit:** B5.3 parity validation needed to forward our small MLX
DiT block AND upstream's PT DiT block on the same inputs and compare.
Problem: upstream's `longcat_video/modules/longcat_video_dit.py` imports
heavily from distributed-training infra that doesn't exist on a single-
GPU / CPU machine:

```python
from ..context_parallel import context_parallel_util
from ..context_parallel.ulysses_wrapper import ulysses_wrapper
from ..block_sparse_attention.bsa_interface import flash_attn_bsa_3d
```

PLUS our installed `longcat_video` package shadows upstream's package
of the same name. Naive `sys.path.insert(0, refs/...)` loads the
wrong module.

**Solution pattern** (now codified in
`tests/parity/test_dit_block_parity.py::upstream_pt_block` fixture):

```python
import sys, types
# 1. Snapshot any installed longcat_video.* modules
saved = {k: v for k, v in sys.modules.items()
         if k == "longcat_video" or k.startswith("longcat_video.")}
for k in list(saved): del sys.modules[k]

# 2. Build a virtual "longcat_video" package rooted at refs/
lc = types.ModuleType("longcat_video"); lc.__path__ = [str(REFS / "longcat_video")]
sys.modules["longcat_video"] = lc

# 3. Inject stubs for distributed-training infra so the upstream code
#    loads. The actual forward path doesn't use any of these — they're
#    only present at import time.
cp = types.ModuleType("longcat_video.context_parallel"); cp.__path__ = []
# (build cp.context_parallel_util / cp.ulysses_wrapper here)
sys.modules["longcat_video.context_parallel"] = cp

# 4. Load the target via importlib.util.spec_from_file_location
spec = importlib.util.spec_from_file_location(
    "longcat_video.modules.longcat_video_dit",
    REFS / "longcat_video" / "modules" / "longcat_video_dit.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# 5. Teardown: restore the snapshot so other tests see our installed pkg
yield mod.LongCatSingleStreamBlock
for k in list(sys.modules):
    if k == "longcat_video" or k.startswith("longcat_video."):
        del sys.modules[k]
sys.modules.update(saved)
```

ALSO: pre-import the MLX equivalent at module-level **before** the
fixture runs. Otherwise the test function's `from longcat_video.models...`
resolves to upstream's virtual package, which doesn't have a `.models.`
subpath.

ALSO: upstream's attention dispatch raises `RuntimeError("Unsupported
attention operations.")` when none of flash-attn / xformers / BSA is
available (i.e., CPU). Monkey-patch `_process_attn` (visual self-attn)
and `forward` (cross-attn — its dispatch is inline) with `torch.nn.
functional.scaled_dot_product_attention` for the test.

ALSO: upstream's RoPE module unconditionally subscripts `cp_split_hw[0]`
— pass `cp_split_hw=[1, 1]` to the block constructor to work around
this single-GPU latent bug.

**Rule for next port:** When PT-side parity needs upstream code that
has distributed-training entanglement:

1. **Don't** add upstream as a path-style import — it WILL collide with
   your installed package
2. **Snapshot + restore `sys.modules`** for the colliding namespace
3. **Pre-import** your MLX equivalent at module level
4. **Stub** distributed-training infra to no-op modules
5. **Monkey-patch** any CUDA-only attention dispatches to PT's SDPA
6. **Catch** known upstream single-GPU latent bugs (e.g.
   `cp_split_hw=None`) by passing safe defaults

Result for this port: DiT block parity **8.94e-08 max_abs** on fp32
CPU stream — bit-for-bit equivalent at the random-init small-block
scale.

**Skill update target:** add to `parity-testing.md` as the canonical
pattern for "PT module needs distributed-training stubs to load."
Tagged toolkit-candidate because every port that wants single-block
PT parity hits this exact problem.

**Artifacts:**
- `tests/parity/test_dit_block_parity.py::upstream_pt_block`
- `tests/parity/_helpers.py` (Avatar pattern)

---

## L48. Parity numbers travel; capture them in a baseline doc

**What we hit:** B5.3 produced concrete max_abs numbers for the DiT
block (8.94e-08), VAE encode (8.05e-06), VAE decode (1.17e-02), umT5
keymap (✅), DiT keymap (✅, 1022 keys). These numbers are the
**ground truth** for future regression detection — if the next change
moves the DiT block from 8.94e-08 to 1e-4, that's a 10,000× regression
even though "still passing." But without writing them down, that
regression is invisible.

The solution: a `docs/development/parity-baseline.md` that records
each test, its threshold, its current value, and the reproduce
command. Update the file whenever the numbers shift. Treat it as a
**checked-in benchmark**.

**Rule for next port:** Don't just write `assert max_abs < threshold`
and walk away. Record the **achieved** max_abs in a baseline file,
along with mean_abs and rel_err. The threshold is the failure trip
wire; the baseline is the regression detector.

When a test starts failing, the diff between baseline and current run
points at the divergent op much faster than re-running with print
statements.

**Skill update target:** add to `parity-testing.md` as
"Record baseline numbers, not just thresholds." Standard practice
for benchmark suites; under-applied for parity suites.

**Artifact:** `docs/development/parity-baseline.md`.

---

## Toolkit candidates (running tally — Avatar + base)

Aggregating tagged lessons for the eventual `mlx-port-toolkit` extraction:

**From Avatar (L1–L22):**
- L4: Conv*d transpose / gamma-skip in safetensors loader
- L7: HF Range-request safetensors header inspection
- The `diag_*.py` bisection template
- The `[parity]` extras pattern
- Smoke/parity split with HF auto-download env var
- L19: HF publish (`repos create --exist-ok` + `upload`)
- L20: Large-repo upload stall handling
- L21: `xcodebuild test` over `swift test` for metallib bundling
- L22: bf16 GPU matmul Python-vs-Swift divergence

**Added in this port (L23–L48):**
- L23: Read published `config.json` before source-spelunking
- L24: Degenerate-case correctness gate (`sparsity=0 ≡ dense`)
- L27 (partial): Fresh-venv release check
- L31 (partial): Separable resampling helper
- L37 (partial): Coarse-to-fine `take` expansion table
- L38: Per-component skip-done sentinels (reinforces Avatar pattern)
- **L39: Merge wrapper must fail loudly on 0-modules-merged**
- L40 (partial): Loader-vs-consumer separation for delta-then-discard
- **L42: Branch parity — every branch must apply every normalization**
  (the stub-DiT-receives-correct-ndim regression test pattern)
- **L43: Quantize from already-converted bf16, don't re-walk PT source**
- **L44: Same `class_predicate` in both recipe and runtime loader**
  (embed `skip_patterns` in published config.json — self-describing)
- **L45: Quant variants reuse non-DiT components verbatim** (only
  quantize what's > 5 GB; copy VAE / umT5 / LoRAs)
- **L47: Run upstream PT via `sys.modules` swap + stubs** (the canonical
  pattern for "PT module needs distributed-training stubs to load on
  single-GPU / CPU")
- L48: Record parity BASELINE numbers, not just thresholds (regression
  detector that catches "still passing but 10,000× worse")

When 3+ ports in a row use the same pattern, that's the extraction trigger.
