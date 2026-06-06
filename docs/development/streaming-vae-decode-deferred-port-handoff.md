# Streaming VAE decode — deferred-port handoff for LongCat (base + avatar)

**Date:** 2026-06-05
**Author of analysis:** Claude (session continuation from lance-mlx PR review arc)
**Status:** Cross-port analysis complete; LongCat audit reveals the
expected port work was largely already done in the original VAE port.
Residual work items identified.
**Read time:** 5 min. Captures why LongCat was deferred behind
phantom-wan + bernini-r, the surprise discovery on audit, and the
small residual scope that remains.

## Why this doc exists

Three substantive optimization PRs landed on `xocialize/lance-mlx`
recently. PR #7 (lossless streaming VAE decode) was identified as
the highest-leverage cross-port opportunity. Initial cross-port analysis
in
[`longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md`](../../../longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md)
queued LongCat for "Stage D — defer to end" with an estimate of 2-4 hr.

User direction: **"defer LongCat to the end either way"** — phantom-wan
+ bernini-r work first.

A second-pass audit of LongCat's own VAE module
(`longcat_video/models/autoencoder_kl_wan.py`, **identical file in
longcat-video-mlx and longcat-avatar-mlx** — both repos share the
vendored copy) before writing this handoff revealed that the streaming
work most of the cross-port analysis assumed was needed had **already
been done in the original VAE port** by following the diffusers
reference implementation closely.

## What's already in place (verified 2026-06-05)

The Lance PR #7 algorithm — temporal causal-cache streaming with
per-frame chunks and a Wan-family `feat_cache`/`feat_idx` plumbing —
is **the default code path** in `AutoencoderKLWan.encode()` and
`AutoencoderKLWan.decode()`. No opt-in flag. No alternate "whole
sequence" mode. Streaming is the only decode path.

### `AutoencoderKLWan.encode()` (autoencoder_kl_wan.py:681-705)

```python
def encode(self, x: mx.array) -> mx.array:
    num_slots = self._count_encoder_cache_slots()
    feat_cache = [None] * num_slots

    t = x.shape[2]
    num_chunks = 1 + (t - 1) // 4

    out = None
    for i in range(num_chunks):
        feat_idx = [0]
        chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
        chunk_out = self.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
        out = chunk_out if out is None else mx.concatenate([out, chunk_out], axis=2)

    mu, _ = mx.split(self.quant_conv(out), 2, axis=1)
    return mu
```

First chunk = 1 frame; subsequent chunks = 4 frames each. Matches Wan2.1
family's temporal stride of 4.

### `AutoencoderKLWan.decode()` (autoencoder_kl_wan.py:716-735)

```python
def decode(self, z: mx.array) -> mx.array:
    x = self.post_quant_conv(z)

    num_slots = self._count_decoder_cache_slots()
    feat_cache: list = [None] * num_slots

    num_frame = x.shape[2]
    out: mx.array | None = None
    for i in range(num_frame):
        feat_idx = [0]
        chunk = x[:, :, i : i + 1]
        chunk_out = self.decoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
        out = chunk_out if out is None else mx.concatenate([out, chunk_out], axis=2)

    return mx.clip(out, -1, 1)
```

Per-latent-frame chunks (chunk_lat=1 effectively hardcoded). `feat_cache`
threaded through each step. Output concatenated along temporal axis.

### `Resample.upsample3d` `"Rep"` sentinel (autoencoder_kl_wan.py:270-310)

Matches diffusers `WanResample.forward` line-for-line: first call marks
`feat_cache[idx] = "Rep"` and skips `time_conv` (correct first-frame
semantics where frame 0 is not doubled); subsequent calls thread
`cache_x` through `time_conv` with cross-chunk state.

### Cache-slot helpers (autoencoder_kl_wan.py:650-680)

`_count_encoder_cache_slots()` and `_count_decoder_cache_slots()` walk
the module tree once at decode time to size the `feat_cache` list.

### Parity test (tests/parity/test_vae_parity.py)

PT vs MLX parity test already in place. Validates the streaming default
against the diffusers reference.

## Why was LongCat deferred if the work was already done?

The deferral itself was correct — but for different reasons than the
original cross-port analysis stated.

### Reason 1 — Multi-port leverage flowed the other way

The headline finding of the cross-port audit was that **mlx-video stock
`wan_2/vae.py`** (the upstream substrate phantom-wan and bernini-r ride
on) is **incomplete for streaming**, despite exposing the API surface:

- `Resample.upsample3d` accepts `feat_cache` / `feat_idx` parameters
  but runs `time_conv` unconditionally with no `"Rep"` sentinel
- `WanVAE.decode()` is whole-sequence — never allocates feat_cache,
  never iterates over chunks
- `decode_tiled` exists but uses lossy trapezoidal-blend (the same
  pattern Lance PR #7 replaced)

So phantom-wan and bernini-r had genuine missing streaming behavior
that needed porting. LongCat already had it. One port effort (against
mlx-video stock, as a consumer-side extension) benefits both
phantom-wan + bernini-r; LongCat needs none of it. **The leverage
ranking placed LongCat last not because the port was harder, but
because there was nothing to port.**

### Reason 2 — User direction

User explicitly directed "defer LongCat to the end either way" before
the audit confirmed the work was already done. Direction was honored
unconditionally; this audit just made the cost of deferral effectively
zero rather than the originally estimated 2-4 hr.

### Reason 3 — File-sharing risk

`autoencoder_kl_wan.py` is **identical** between longcat-video-mlx and
longcat-avatar-mlx (verified via `diff -q`). The base README states:
"80% of this repo's reusable modules were copy-vendored from
[longcat-avatar-mlx]: Wan VAE, umT5-XXL, base DiT … Avatar bug fixes
do not auto-flow here; mirror them manually if they touch shared
modules." Any modification to the VAE has to be applied to both repos
identically and the parity tests re-run in both. Deferring concentrates
that two-repo coordination work into one focused session if/when it
becomes necessary, rather than scattered across phantom-wan/bernini-r
debugging sessions.

## Residual scope (what's actually left to do on LongCat)

The original cross-port analysis estimated 2-4 hr for "port streaming
decode to LongCat." That estimate is now **near zero for the streaming
algorithm itself**, but a small list of residual items remains. Each is
optional.

### Item 1 — Self-consistency bit-identity test (recommended)

LongCat's `tests/parity/test_vae_parity.py` validates MLX vs PyTorch
parity but does NOT (as far as the file inventory shows) assert
self-consistency of the streaming decode — i.e. there's no test that
proves `streaming_decode(z) == whole_sequence_decode(z)` bit-exact for
the MLX implementation alone, weights-free.

Lance has exactly this test: `lance-mlx/tests/test_decode_stream.py`
(280 LOC, weights-free, 50 cases across T_lat ∈ {1, 2, 3, 5, 9}).
It's the regression guard that catches a future "optimization" that
accidentally breaks streaming semantics — the kind of bug that would
not surface in a PT-vs-MLX parity test because PT also runs streaming
(diffusers reference). The bug surfaces only when streaming-with-cache
silently produces different output than streaming-with-fresh-cache.

For LongCat: build a tiny random-init `AutoencoderKLWan` (no weights
needed, just instantiate), assert that two calls — one with chunk_lat=1
(current default) and one synthesized with chunk_lat=full_temporal —
produce bit-identical outputs. The full-temporal "reference" can be
synthesized by running `self.decoder(x)` directly without the per-frame
loop. ~150 LOC of test, ~30 min to write.

**Recommended.** Cheap regression guard against future drift.

### Item 2 — `chunk_lat > 1` opt-in (optional, defer)

Lance's `decode_streaming(z, chunk_lat=1)` accepts a `chunk_lat`
parameter so users can trade memory for throughput. LongCat's `decode()`
hardcodes chunk_lat=1.

The benefit is small on Apple Silicon (LongCat's bottleneck at
production scale is the DiT forward, not VAE decode), but it's a
useful surface for future tuning. ~30 LOC to add as an optional kwarg
defaulting to 1 (preserving current behavior).

**Defer** until someone benchmarks decode as a bottleneck.

### Item 3 — Spatial halo-tile (optional, defer)

Lance's `vae_stream.py` Phase 2 adds a spatial halo-tile pass for very
large frames (>1024² latent → >32k² output). LongCat's typical
production envelope is 720p (output ~1280×720), which doesn't need
spatial tiling — single-frame decode fits comfortably.

**Defer indefinitely** unless someone targets a much larger output
resolution.

### Item 4 — Document the streaming-by-default behavior in the model card

LongCat's HF model cards (when published) don't mention that the VAE
decode is streaming-by-default. Worth a one-line note for users who
might wonder why their long-video decodes don't grow linearly in
memory the way they might expect from a naive PyTorch reference.

**Recommended after publish.** ~5 min of writing.

### Item 5 — Cross-port leverage flow: nothing additional

There's no additional cross-port work to flow from LongCat *out* — the
streaming pattern is already in mlx-video stock as of Stage A's work
(if/when it lands). Future ports inherit it from upstream rather than
from LongCat.

## Updated cross-port sequence

Reflecting this audit:

1. **Stage A** — consumer-side streaming extension in
   `phantom_wan_mlx/streaming_decode.py` (~3-5 hr, against mlx-video
   stock `wan_2/vae.py`). See
   [`phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md`](../../../phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md).
2. **Stage B** — phantom-wan pipeline wires `lossless_decode=True`
   (absorbed into Stage A).
3. **Stage C** — bernini-r mirrors (~30 min, copy file).
4. **Stage D — REVISED** — LongCat audit (this doc): NO port needed.
   Optional ~30 min for the self-consistency bit-identity test
   (Item 1) when a focused session opens.

## What this means for the LongCat-avatar cross-port analysis

The doc
[`longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md`](../../../longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md)
currently states (revised 2026-06-05):

> Effort (revised 2026-06-05): Cheaper than original estimate.
> Our `Resample.upsample3d` already has the diffusers `"Rep"` sentinel
> cache pattern, so the per-stage cross-chunk plumbing exists. Only the
> top-level orchestrator (`AutoencoderKLWan.decode_streaming(z,
> chunk_lat=1)` that allocates `feat_cache = [None] * num_slots` and
> walks z by temporal chunks) is missing. Maybe ~150-200 LOC + 250 LOC
> test = 2-4 hours.

That sentence is **wrong** — the orchestrator IS already in place at
`AutoencoderKLWan.decode()` (it just isn't named `decode_streaming`
because there's no whole-sequence alternative to distinguish from). The
cross-port doc should be updated to reflect: PR #7's algorithm is
already shipped as the default decode path; residual scope is one
optional bit-identity test (~30 min).

(This handoff doc supersedes the relevant section in that file. The
correction will be folded into that file when LongCat work is next
touched.)

## Why this is a useful lesson for future cross-port analyses

The pattern of getting cross-port effort estimation wrong on a fork
that already implemented the optimization is worth flagging — this
session has now hit it twice:

1. **First-pass estimate (this session, before the avatar audit):**
   "Port to LongCat = 4-8 hr; we need to add the Rep sentinel and the
   orchestrator." Wrong — the Rep sentinel was already there.
2. **Second-pass estimate (this session, after avatar audit but before
   base audit):** "Port to LongCat = 2-4 hr; just add the orchestrator."
   Also wrong — the orchestrator was already there too.
3. **Third-pass (this audit):** "Port to LongCat = 0 hr; it's already
   done. Maybe 30 min for an optional regression test."

The reason for the misses: I read LongCat's `Resample.upsample3d` to
verify the Rep sentinel and saw the streaming primitive was wired, but
didn't audit the top-level `AutoencoderKLWan.decode()` until I had a
concrete reason to (writing this handoff). The skill-level lesson
(mlx-porting pitfall #15) was about consumer-side extension —
the *complement* lesson is: **before estimating port effort to a
target fork, audit the target's top-level entry point, not just the
inner blocks.** A fork that followed the reference implementation
closely often already has the capability you're considering porting in.

This is worth folding into the mlx-porting skill as a one-line
addition to pitfall #15 or as a new pitfall #16 if it recurs in a
future port.

## Concrete pickup if/when LongCat work resumes

When a focused session opens:

1. **Verify this audit holds.** Re-read
   `longcat_video/models/autoencoder_kl_wan.py:716-735`. Confirm
   `decode()` still implements the per-frame streaming loop with
   `feat_cache` threading. Same for `encode()` at line 681. If
   either was refactored, re-assess.
2. **Write the self-consistency bit-identity test** (Item 1). ~30 min,
   ~150 LOC, weights-free. Land in both
   `longcat-video-mlx/tests/parity/` and
   `longcat-avatar-mlx/tests/parity/` (same VAE = same test mirrored).
   Confirm it passes against current code.
3. **Update the LongCat-avatar cross-port analysis** to mark PR #7 as
   "audit complete, residual = 1 optional test, not a port."
4. **Update HF model cards** (Item 4) on publish to mention
   streaming-by-default behavior.
5. **Done.** Total work ~1 hr including documentation.

## Reference files

**The current LongCat VAE port (verified streaming-by-default):**
- `longcat-video-mlx/longcat_video/models/autoencoder_kl_wan.py`
  (identical to)
- `longcat-avatar-mlx/longcat_video_avatar/models/autoencoder_kl_wan.py`

**The Lance algorithm + test reference:**
- `lance-mlx/src/lance_mlx/model/vae_stream.py` (426 LOC)
- `lance-mlx/tests/test_decode_stream.py` (280 LOC) — gold-standard
  pattern for the self-consistency test

**Cross-port analysis (currently overstates LongCat scope):**
- `longcat-avatar-mlx/docs/development/lance-mlx-pr-cross-port-analysis.md`

**Phantom-Wan handoff (the upstream-substrate port that's actually
needed):**
- `phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md`

**Original schema-mismatch note that explains why LongCat forked:**
- `longcat-avatar-mlx/docs/development/notes/vae-schema-mismatch.md`
  (still accurate — channel arithmetic at `dim_mult=[1,2,4,4]` boundary
  still prevents un-forking onto mlx-video stock)
