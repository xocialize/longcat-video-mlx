# BSA Tier B — Metal kernel design

## Key benchmark finding (B4.1a)

Tier A pure-MLX BSA is actually **2-3× SLOWER than dense SDPA** at
typical sequence lengths:

| Shape | Tokens | Heads | dense SDPA | Tier A BSA | Ratio |
|---|---|---|---|---|---|
| (T=4, H_lat=8, W_lat=12) | 384 | 8 | 0.54 ms | 0.96 ms | 1.78× slower |
| (T=8, H_lat=16, W_lat=16) | 2048 | 16 | 0.95 ms | 2.84 ms | 2.98× slower |
| (T=4, H_lat=16, W_lat=20) | 1280 | 16 | 0.55 ms | 1.30 ms | 2.35× slower |

This is because Tier A:
1. Mean-pools Q/K into block representatives
2. Computes block-level routing scores
3. Takes top-k indices
4. Materializes the **full S²** token-pair mask via `mx.take`
5. Dispatches dense SDPA with the additive mask

Steps 1-3 are cheap. Step 4 dominates — at S=2048, the mask is 4M bools
× 16 heads = 64M bytes of writes that immediately serve as a 256 MB
additive-mask tensor. Step 5 does the same work as dense + the mask add.

**Implication for Tier B:** the bar is "beat dense at refinement-pass
shapes (S > 10K typically)," not just "beat Tier A." At small S, dense
is already fast enough — the perf-relevant shapes are large.

## Refinement-pass target shapes

At 720p (1280×720) the latent is roughly (T_lat=8, H_lat=90, W_lat=160)
after VAE encode. Visual seq:

  S = T_lat × H_lat × W_lat = 8 × 90 × 160 = 115,200 tokens

Per attention head: ~115K × 128 = 14.7M params. 16-32 heads.

At BSA sparsity 0.9375 (keep 6.25%):
- 64-token blocks → 1800 blocks
- Each Q block attends to 113 KV blocks ≈ 7232 tokens

Dense FLOPs per layer: 2 × 115,200² × 128 ≈ 3.4 TFLOP per head, × 32
heads = 110 TFLOP. Apple M2 Pro = 6.8 TFLOPs sustained → ~16 sec/layer
for dense. **Untenable** for 48 layers × ~25 refinement steps.

Tier B target: keep the 6.25% sparsity, hit ~700 GFLOPs per layer × 48
× 25 = 0.84 PFLOP total → ~2 min refinement on M2 Pro. **Acceptable**.

## Design — naive correct kernel first (Phase 1)

Build the simplest possible correct kernel:

**Inputs:**
- Q, K, V: `[B, H, S, D]` fp16
- block_indices: `[B, H, num_q_blocks, top_k]` int32 — output of Tier A's
  `topk_block_indices`. Tier B does NOT re-do routing — it consumes the
  same routing decisions as Tier A so the correctness invariant L24
  (`sparsity=0 ≡ dense`) extends naturally.
- block_size (template): 64
- D (template): typically 128
- top_k (template): integer

**Output:** O: `[B, H, S, D]` fp16

**Algorithm (one thread per output token):**

```
thread_id maps to (b, h, q_global_idx) where q_global_idx = 0..S-1.
q_block = q_global_idx / block_size
q_offset_in_block = q_global_idx % block_size

Load Q[b, h, q_global_idx, :] into thread-local D-dim vector (registers).

Initialize: m = -inf, l = 0, o = zeros(D)   // online softmax state

For k in 0..top_k:
    kv_block_idx = block_indices[b, h, q_block, k]
    
    For j in 0..block_size:
        kv_global_idx = kv_block_idx * block_size + j
        
        // Compute score
        s = (Q · K[b, h, kv_global_idx, :]) * scale
        
        // Online softmax update
        m_new = max(m, s)
        l_new = l * exp(m - m_new) + exp(s - m_new)
        o     = o * exp(m - m_new) + exp(s - m_new) * V[b, h, kv_global_idx, :]
        
        m = m_new; l = l_new

Output[b, h, q_global_idx, :] = o / l
```

**Threadgroup geometry:**
- Grid: `(S, H, B)` — total threads = B × H × S (one per output token)
- Threadgroup: `(32, 1, 1)` — one simdgroup per group, gives the GPU
  scheduler flexibility

**No threadgroup memory** in Phase 1. Each thread does its own reads
from global memory. This is **memory-bandwidth-bound** rather than
**compute-bound** — likely sub-optimal but simple and correct.

**Memory pattern per thread:**
- Reads Q once: D fp16 = 256 bytes
- For each of (top_k × block_size) attended tokens: 2D fp16 reads (K and
  V) = 512 bytes × ~7000 = ~3.5 MB per thread

Apple Silicon GPU L1 cache is ~32 KB per core; 3.5 MB will spill. So
this kernel will be bandwidth-bound on the K/V reads. Reuse via
threadgroup shared memory is the obvious optimization for Phase 2.

## Phase 2 (if Phase 1 isn't fast enough)

- **Threadgroup-shared Q**: threads in same threadgroup share the same
  Q block. Load Q[q_block] into 16 KB shared memory once per threadgroup.
- **Tile KV blocks through shared memory**: load a KV block into shared,
  have all threads compute their attention slice against it, then
  advance. This is the canonical FlashAttention-2 pattern.
- **Per-warp tiles** (simdgroup_matrix): use Metal's
  `simdgroup_matrix<T, 8, 8>` for 8×8 fp16 matmul tiles. Native HW
  acceleration.

## Correctness gates

1. **L24 degenerate case**: `bsa_tier_b(sparsity=0) ≡ dense SDPA`
   (drift < 1e-4 fp16). This is the strongest possible correctness test.
2. **Tier B ≡ Tier A** for non-degenerate sparsity: both compute the
   same routed attention, so outputs should match within fp16 noise.
3. **Random sparsity sweep**: sparsity in {0.1, 0.5, 0.9, 0.9375} — all
   should match Tier A within ~1e-3.

## Risks / open questions

- **Metal threadgroup memory limit**: 32 KB on M-series. Q block (64×128×fp16 = 16 KB) + 1 KV block (16 KB) = 32 KB exactly — tight.
- **JIT compile time**: `mx.fast.metal_kernel` first-call may take seconds. The plan v1 mitigation: pre-warm at pipeline initialization. We'll do this in `enable_bsa()`.
- **Fp16 vs fp32 accumulation**: softmax max + sum accumulators need fp32 stability; output reduction may need fp32. Investigate per-Phase.
- **Per-block index encoding**: block_indices shape `[B, H, num_q_blocks, top_k]` int32 — total ~9 MB per layer at production shapes. Acceptable.

## Phase 1 results (B4.1c)

**Built it. It works. It's correct but slow.**

Correctness ✅:
- Degenerate case (sparsity=0): Tier B ≡ dense SDPA within 5e-3 (fp16
  Metal-GPU tolerance per L11)
- No-sparsity multi-block: Tier B ≡ Tier A within 5e-3
- Partial sparsity 0.5: Tier B ≡ Tier A within 5e-3
- Rejects misaligned S (refinement padding invariant)
- 6/6 smoke tests pass

Performance (fp16, M-series, sparsity=0.9375):

| shape | dense | Tier A | **Tier B** | B vs dense | B vs A |
|---|---|---|---|---|---|
| small (S=384) | 0.28 ms | 0.52 ms | 0.61 ms | **0.46×** | 0.85× |
| medium (S=2048) | 0.97 ms | 2.86 ms | 6.20 ms | **0.16×** | 0.46× |
| wide (S=1280) | 0.53 ms | 1.34 ms | 2.49 ms | **0.21×** | 0.54× |
| large (S=3840) | 2.67 ms | 8.74 ms | **19.54 ms** | **0.14×** | 0.45× |

Phase 1 is **2-7× slower than dense SDPA** at typical shapes. Exactly
what the design doc predicted for a naive single-thread-per-output-token
kernel without threadgroup memory or simdgroup_matrix HW acceleration.

## Why Phase 1 is slow

1. **No K/V reuse across threads.** Each thread reads its 7000 KV tokens
   from global memory independently. L1 cache (~32 KB) overflows after
   ~256 KV tokens at D=128. We re-fetch from L2/HBM constantly.
2. **No HW-accelerated matmul.** `dense SDPA` uses `simdgroup_matrix`
   primitives that do 8×8 fp16 matmul in a single op. Our naive kernel
   does scalar fp32 multiply-adds in a loop.
3. **Memory access pattern is bad.** Each thread strides through Q/K/V
   non-uniformly; the GPU prefers coalesced reads where adjacent threads
   read adjacent memory.

## Phase 2 design notes (what would close the perf gap)

To **beat dense SDPA** at refinement-pass shapes:

1. **Shared-Q tile**: load each Q block (64×128 fp16 = 16 KB) into
   threadgroup memory once per group. All threads in the group share it.
2. **Streamed KV tiles**: load one KV block at a time into 16 KB of
   shared memory. Compute partial attention for ALL Q rows against this
   tile. Advance. This is FlashAttention-2's "outer loop over K, inner
   over Q" structure.
3. **`simdgroup_matrix<fp16, 8, 8>`**: use Metal's HW-accelerated fp16
   matmul primitive. One simdgroup (32 threads) computes an 8×8 fp16
   matmul block in a single op. For a 64×128 Q × 128×64 K^T = 64×64
   score matrix, that's 8×8 = 64 simdgroup_matrix ops per tile.
4. **Per-row fp32 softmax accumulators kept in registers**: each thread
   owns 2 output rows. Running max/sum/output stay in registers across
   KV tiles.

**Estimated Phase 2 speedup:** 5-10× faster than dense at S>10K (the
refinement-pass regime). This matches the Triton CUDA kernel's reported
~8× speedup over dense.

**Estimated Phase 2 effort:** 3-5 days of focused Metal kernel work +
testing. Substantial.

## Recommendation

**Phase 1 ships as the "correct, opt-in" path.** Default to Tier A in
the refinement pipeline (faster than Phase 1 at all current shapes).
Expose Tier B as opt-in for users running Phase 2 work.

**Phase 2 is a substantial Metal optimization sprint.** Deep research
on:
- `simdgroup_matrix<T, M, N>` usage patterns in production MLX kernels
- FlashAttention-2 paper's exact tile-sizing math for 32 KB threadgroup
  budget on Apple Silicon
- Any community/published BSA Metal kernels (Triton, FlashAttention,
  FlexAttention) that could be adapted

…would substantially de-risk Phase 2.
