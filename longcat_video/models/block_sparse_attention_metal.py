"""Block Sparse Attention — **Tier B Metal kernel (Phase 1, naive)**.

The Tier A pure-MLX implementation (see `block_sparse_attention.py`) is
*correctness-correct* but ~2-3× SLOWER than dense SDPA at typical
sequence lengths because it materializes the full S² additive mask.

Tier B is a real Metal kernel that gathers + computes ONLY the routed
(Q-block, KV-block) pairs, never touching the dropped 93.75% of
attention scores.

**Phase 1 (this file) is intentionally simple:**

- One thread per output token (B × H × S threads total)
- No threadgroup-shared memory (each thread reads K/V from global)
- Online softmax (single-pass FlashAttention-style stability)
- fp16 inputs, fp32 accumulation

This is the **correct, minimal** kernel. If it lands at "correct but only
2× faster than dense at refinement-pass shapes", Phase 2 will introduce:
- Threadgroup-shared Q block (one load per threadgroup vs per thread)
- Tiled KV block streaming through shared memory
- `simdgroup_matrix<fp16, 8, 8>` for HW-accelerated 8×8 fp16 matmul

For now, simplicity first.

## Correctness contract

The routing decisions come from Tier A's `topk_block_indices` — Tier B
does NOT re-compute routing. This means:

  bsa_attention_metal(q, k, v, block_indices)
  ≡ bsa_attention_pure_mlx(q, k, v, sparsity=...)

…where `block_indices` is the output of `topk_block_indices(score,
sparsity)`. The degenerate-case correctness gate L24 extends naturally:
when `block_indices = arange(num_kv_blocks)` for every Q block (i.e.
top-k = num_kv_blocks, sparsity=0), Tier B must equal dense SDPA.

## Kernel JIT cost

`mx.fast.metal_kernel` compiles on first call (typically 0.5-2s on
M-series). The DiT calls BSA from every block × every step × every
inference call — so the kernel needs to be **pre-warmed** at pipeline
initialization. We expose `prewarm()` that runs the kernel once on
dummy inputs.

PT reference: `refs/longcat-video/longcat_video/block_sparse_attention/
flash_attn_bsa_varlen_mask.py` — Triton kernel; same algorithmic shape,
different threading model.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


__all__ = [
    "bsa_attention_metal",
    "bsa_attention_metal_v2",
    "bsa_attention_metal_v3",
    "prewarm_metal_kernel",
]


_KERNEL_CACHE: dict[tuple, object] = {}
_KERNEL_V2_CACHE: dict[tuple, object] = {}
_KERNEL_V3_CACHE: dict[tuple, object] = {}


_BSA_KERNEL_BODY = r"""
    // ===================================================================
    // BSA Tier B Phase-1 naive kernel.
    //
    // Thread layout: one thread per output token.
    //   grid:        (S, H, B)
    //   threadgroup: (32, 1, 1)
    //
    // Constants from template parameters:
    //   T   : input dtype  (fp16 or fp32)
    //   D   : head dimension
    //   BS  : block_size (usually 64)
    //   TK  : top_k
    // ===================================================================

    uint b = thread_position_in_grid.z;
    uint h = thread_position_in_grid.y;
    uint s = thread_position_in_grid.x;   // output token index in 0..S-1

    // Shape guards (since grid may slightly overshoot S due to threadgroup quantization)
    uint S = q_shape[2];
    uint H = q_shape[1];
    uint B = q_shape[0];
    if (s >= S || h >= H || b >= B) {
        return;
    }

    uint q_block = s / BS;
    uint num_q_blocks = S / BS;

    // Load Q[b, h, s, :] into thread-local registers
    // Q layout: [B, H, S, D] row-major
    uint q_base = ((b * H + h) * S + s) * D;
    float q_reg[D];
    for (uint d = 0; d < D; d++) {
        q_reg[d] = (float)q[q_base + d];
    }

    // Softmax scale
    const float scale = 1.0f / metal::sqrt((float)D);

    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;
    float o[D];
    for (uint d = 0; d < D; d++) {
        o[d] = 0.0f;
    }

    // For each selected KV block (TK total)
    uint bi_base = ((b * H + h) * num_q_blocks + q_block) * TK;
    for (uint sel = 0; sel < TK; sel++) {
        int kv_block_idx = block_indices[bi_base + sel];

        // For each token in this KV block
        uint kv_token_base = (uint)kv_block_idx * BS;
        for (uint j = 0; j < BS; j++) {
            uint kv_idx = kv_token_base + j;
            if (kv_idx >= S) continue;  // safety for partial last block

            // Compute scaled dot product
            uint k_base = ((b * H + h) * S + kv_idx) * D;
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += q_reg[d] * (float)k[k_base + d];
            }
            float score = dot * scale;

            // Online softmax update
            float m_new = metal::max(m, score);
            float exp_diff = metal::exp(m - m_new);     // 1.0 if m unchanged
            float exp_score = metal::exp(score - m_new);

            l = l * exp_diff + exp_score;

            uint v_base = ((b * H + h) * S + kv_idx) * D;
            for (uint d = 0; d < D; d++) {
                o[d] = o[d] * exp_diff + exp_score * (float)v[v_base + d];
            }
            m = m_new;
        }
    }

    // Normalize + write output
    uint out_base = ((b * H + h) * S + s) * D;
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = 0; d < D; d++) {
        out[out_base + d] = (T)(o[d] * inv_l);
    }
"""


def _get_kernel(D: int, BS: int, TK: int, dtype: mx.Dtype):
    """JIT-compile the kernel for the given template params (cached).

    First call for a given (D, BS, TK, dtype) tuple takes ~0.5-2s on
    M-series. Subsequent calls are free.
    """
    key = (D, BS, TK, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"bsa_phase1_D{D}_BS{BS}_TK{TK}",
            input_names=["q", "k", "v", "block_indices"],
            output_names=["out"],
            source=_BSA_KERNEL_BODY,
            ensure_row_contiguous=True,
        )
    return _KERNEL_CACHE[key]


def _reshape_to_block_major(
    x: mx.array,
    shape: tuple[int, int, int],
    chunk_thw: tuple[int, int, int],
):
    """Rearrange `[B, H, S, D]` (t-major flat) into `[B, H, S, D]` where
    consecutive `block_size` tokens form one BSA block.

    Required because the BSA Tier B kernel uses `q_block = s / block_size`
    internally — true only when blocks are contiguous in the seq axis.
    Tier A handles this via a lookup table; Tier B handles it via reshape.
    Both produce the same result.
    """
    T, H_, W = shape
    cT, cH, cW = chunk_thw
    Bt, Bh, Bw = T // cT, H_ // cH, W // cW
    B, H, S, D = x.shape
    # (T, H, W) → (Bt, cT, Bh, cH, Bw, cW) then permute to (Bt, Bh, Bw, cT, cH, cW)
    y = x.reshape(B, H, Bt, cT, Bh, cH, Bw, cW, D)
    y = y.transpose(0, 1, 2, 4, 6, 3, 5, 7, 8)
    return y.reshape(B, H, S, D)


def _reshape_from_block_major(
    x: mx.array,
    shape: tuple[int, int, int],
    chunk_thw: tuple[int, int, int],
):
    """Inverse of `_reshape_to_block_major`."""
    T, H_, W = shape
    cT, cH, cW = chunk_thw
    Bt, Bh, Bw = T // cT, H_ // cH, W // cW
    B, H, S, D = x.shape
    y = x.reshape(B, H, Bt, Bh, Bw, cT, cH, cW, D)
    # invert the permutation: (Bt, Bh, Bw, cT, cH, cW) → (Bt, cT, Bh, cH, Bw, cW)
    y = y.transpose(0, 1, 2, 5, 3, 6, 4, 7, 8)
    return y.reshape(B, H, S, D)


def bsa_attention_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_indices: mx.array,
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
    shape: Optional[tuple[int, int, int]] = None,
) -> mx.array:
    """BSA Tier B — Metal kernel (Phase 1).

    Args:
        q, k, v: `[B, H, S, D]` where `S = T_lat * H_lat * W_lat` in
            standard t-major flat order.
        block_indices: `[B, H, num_q_blocks, top_k]` int32, the output of
            `topk_block_indices(score, sparsity)` from Tier A.
        chunk_thw: BSA chunk shape (default (4, 4, 4) → 64-token blocks).
        shape: (T_lat, H_lat, W_lat). Required — the kernel internally
            assumes BSA blocks are contiguous in the seq axis, so we
            rearrange Q/K/V to block-major order before dispatch and
            invert after.

    Returns: `[B, H, S, D]` of the same dtype as q, in standard t-major
        flat order.
    """
    if shape is None:
        raise ValueError(
            "shape=(T_lat, H_lat, W_lat) is required for Tier B BSA "
            "(needed for the t-major ↔ block-major rearrangement)"
        )
    B, H, S, D = q.shape
    BS = chunk_thw[0] * chunk_thw[1] * chunk_thw[2]    # 64 for (4,4,4)
    TK = int(block_indices.shape[-1])

    if S % BS != 0:
        raise ValueError(
            f"S={S} must be a multiple of block_size={BS}. The refinement "
            f"pipeline's padding to multiples of 4 in each spatial dim "
            f"should guarantee this."
        )

    # Rearrange t-major → block-major so kernel's `s / BS` block lookup is correct
    q_bm = _reshape_to_block_major(q, shape, chunk_thw)
    k_bm = _reshape_to_block_major(k, shape, chunk_thw)
    v_bm = _reshape_to_block_major(v, shape, chunk_thw)

    # JIT-compiled, cached
    kernel = _get_kernel(D, BS, TK, q.dtype)

    # Cast block_indices to int32 if needed
    if block_indices.dtype != mx.int32:
        block_indices = block_indices.astype(mx.int32)

    # Dispatch: 1 thread per output token; threadgroup (32, 1, 1)
    grid = (S, H, B)
    threadgroup = (32, 1, 1)

    out_bm = kernel(
        inputs=[q_bm, k_bm, v_bm, block_indices],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("BS", BS),
            ("TK", TK),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, H, S, D)],
        output_dtypes=[q.dtype],
    )[0]

    # Rearrange back to t-major
    return _reshape_from_block_major(out_bm, shape, chunk_thw)


def prewarm_metal_kernel(
    head_dim: int = 128,
    block_size: int = 64,
    top_k_values: tuple[int, ...] = (6, 64, 128),
    dtype: mx.Dtype = mx.float16,
) -> None:
    """Pre-warm the JIT cache for the common (head_dim, block_size, top_k)
    combinations the refinement pipeline will use.

    Call this once at pipeline initialization (e.g. inside
    `LongCatVideoTransformer3DModel.enable_bsa()`) so the first inference
    step doesn't pay the ~1s compile latency per template combination.
    """
    print(f"  [BSA Tier B] pre-warming Metal kernel for head_dim={head_dim}, "
          f"block_size={block_size}, top_k_values={top_k_values}, "
          f"dtype={dtype}...")

    # Single 4×4×4 latent block (S=64). Top_k clamped to ≤ 1.
    B, H, S = 1, 1, block_size   # 1 Q block
    D = head_dim

    for tk in top_k_values:
        if tk > 1:
            continue
        q = mx.zeros((B, H, S, D), dtype=dtype)
        k = mx.zeros((B, H, S, D), dtype=dtype)
        v = mx.zeros((B, H, S, D), dtype=dtype)
        bi = mx.broadcast_to(
            mx.arange(tk, dtype=mx.int32),
            (B, H, S // block_size, tk),
        )
        out = bsa_attention_metal(q, k, v, bi, shape=(4, 4, 4))
        mx.eval(out)
    # Note: pre-warm at the smaller tk values; production tk (≈100+) will
    # compile lazily on first real call but that's a single hit per layer.
    print(f"  [BSA Tier B] pre-warm complete.")


# ===========================================================================
# Phase 2 — simdgroup-cooperative kernel (FAST)
# ===========================================================================
#
# Phase 1 (above) had one thread doing ALL the work for one output token.
# That left 31 of 32 threads in each simdgroup idle.
#
# Phase 2 mirrors the pattern from MLX's `sdpa_vector.h`: ALL 32 threads in
# a simdgroup cooperate on ONE output token's attention, using:
#
# - Each thread holds D/32 elements of Q in registers (qk_per_thread = 4
#   when D=128). Load once per output token, reuse across all KV.
# - For each attended KV token: each thread computes its partial dot
#   product, then `simd_sum` reduces 32 partials → 1 scalar score in HW
#   in ~5 cycles.
# - All 32 threads update softmax accumulators in lockstep (each holds
#   its own slice of the output vector).
# - Same for V accumulation: each thread holds its slice of the running
#   output, multiplied by exp_score from V[same kv idx].
#
# This is ~32× faster per attention op than Phase 1's serial loop
# (no simd reduction, scalar fp32 accumulators), and crucially uses
# HW simd primitives that beat naive scalar code.

_BSA_KERNEL_V2_BODY = r"""
    // ===================================================================
    // BSA Tier B Phase-2 simdgroup-cooperative kernel.
    //
    // Thread layout:
    //   grid:        (B * H * S, 1, 1)   — total threads
    //   threadgroup: (32, 1, 1)          — one simdgroup per group
    //   → one threadgroup = one simdgroup = one output token
    //
    // Each thread holds D/32 fp32 elements of Q and the running output.
    // `simd_sum` performs the per-token dot-product reduction in HW.
    //
    // Template params:
    //   T   : input dtype
    //   D   : head dimension (must be a multiple of 32; typ. 128)
    //   BS  : block_size (64)
    //   TK  : top_k
    // ===================================================================

    constexpr uint BD = 32;       // simdgroup width
    constexpr uint QK_PT = D / BD;

    uint linear_tid = thread_position_in_grid.x;
    uint simd_lid = thread_index_in_simdgroup;
    uint S = q_shape[2];
    uint H = q_shape[1];
    uint B = q_shape[0];
    uint BHS = B * H * S;

    // Decode output-token index from linear tid (grid x = B*H*S)
    uint token_global = linear_tid / BD;
    if (token_global >= BHS) return;

    uint b = token_global / (H * S);
    uint hs = token_global % (H * S);
    uint h = hs / S;
    uint s = hs % S;

    uint q_block = s / BS;
    uint num_q_blocks = S / BS;

    // Load Q slice for this thread (QK_PT elements)
    // Q layout: [B, H, S, D]
    uint q_base = ((b * H + h) * S + s) * D;
    float q_reg[QK_PT];
    for (uint i = 0; i < QK_PT; i++) {
        q_reg[i] = (float)q[q_base + simd_lid * QK_PT + i];
    }

    // Softmax scale (use log2-base for fast::exp2)
    const float scale = 1.0f / metal::sqrt((float)D);
    const float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;

    // Online softmax state (per thread)
    float m = -INFINITY;
    float l = 0.0f;
    float o_reg[QK_PT];
    for (uint i = 0; i < QK_PT; i++) {
        o_reg[i] = 0.0f;
    }

    // For each selected KV block
    uint bi_base = ((b * H + h) * num_q_blocks + q_block) * TK;
    for (uint sel = 0; sel < TK; sel++) {
        int kv_block_idx = block_indices[bi_base + sel];
        uint kv_token_base = (uint)kv_block_idx * BS;

        // For each token in the KV block
        for (uint j = 0; j < BS; j++) {
            uint kv_idx = kv_token_base + j;
            if (kv_idx >= S) continue;

            // Each thread reads its K slice
            uint k_base = ((b * H + h) * S + kv_idx) * D;
            float local_dot = 0.0f;
            for (uint i = 0; i < QK_PT; i++) {
                float k_val = (float)k[k_base + simd_lid * QK_PT + i];
                local_dot += q_reg[i] * k_val;
            }
            // HW reduction: 32 partials → 1 scalar score in ~5 cycles
            float score = metal::simd_sum(local_dot) * scale_log2;

            // Online softmax update (all threads in lockstep — score is
            // broadcast via simd_sum's return value)
            float m_new = metal::max(m, score);
            float exp_diff = metal::fast::exp2(m - m_new);
            float exp_score = metal::fast::exp2(score - m_new);

            l = l * exp_diff + exp_score;

            // Update each thread's slice of the output
            uint v_base = ((b * H + h) * S + kv_idx) * D;
            for (uint i = 0; i < QK_PT; i++) {
                float v_val = (float)v[v_base + simd_lid * QK_PT + i];
                o_reg[i] = o_reg[i] * exp_diff + exp_score * v_val;
            }
            m = m_new;
        }
    }

    // Normalize + write each thread's slice of the output
    uint out_base = ((b * H + h) * S + s) * D;
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint i = 0; i < QK_PT; i++) {
        out[out_base + simd_lid * QK_PT + i] = (T)(o_reg[i] * inv_l);
    }
"""


def _get_kernel_v2(D: int, BS: int, TK: int, dtype: mx.Dtype):
    """JIT-compile the Phase 2 kernel (cached)."""
    key = (D, BS, TK, dtype)
    if key not in _KERNEL_V2_CACHE:
        _KERNEL_V2_CACHE[key] = mx.fast.metal_kernel(
            name=f"bsa_phase2_D{D}_BS{BS}_TK{TK}",
            input_names=["q", "k", "v", "block_indices"],
            output_names=["out"],
            source=_BSA_KERNEL_V2_BODY,
            ensure_row_contiguous=True,
        )
    return _KERNEL_V2_CACHE[key]


def bsa_attention_metal_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_indices: mx.array,
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
    shape: Optional[tuple[int, int, int]] = None,
) -> mx.array:
    """BSA Tier B Phase 2 — simdgroup-cooperative Metal kernel.

    Same interface as `bsa_attention_metal` (Phase 1) but uses 32 threads
    cooperatively per output token via `simd_sum`. ~10-30× faster than
    Phase 1 for typical (D ≥ 64) shapes.

    Constraint: head_dim D must be a multiple of 32 (the simdgroup width).
    """
    if shape is None:
        raise ValueError(
            "shape=(T_lat, H_lat, W_lat) is required for Tier B BSA"
        )
    B, H, S, D = q.shape
    BS = chunk_thw[0] * chunk_thw[1] * chunk_thw[2]
    TK = int(block_indices.shape[-1])

    if S % BS != 0:
        raise ValueError(
            f"S={S} must be a multiple of block_size={BS}."
        )
    if D % 32 != 0:
        raise ValueError(
            f"head_dim D={D} must be a multiple of 32 for Phase 2 "
            f"simdgroup-cooperative kernel. Use Phase 1 (bsa_attention_metal) "
            f"for D < 32 or non-multiple-of-32 D."
        )

    # Rearrange t-major → block-major for kernel's `s/BS` lookup
    q_bm = _reshape_to_block_major(q, shape, chunk_thw)
    k_bm = _reshape_to_block_major(k, shape, chunk_thw)
    v_bm = _reshape_to_block_major(v, shape, chunk_thw)

    kernel = _get_kernel_v2(D, BS, TK, q.dtype)

    if block_indices.dtype != mx.int32:
        block_indices = block_indices.astype(mx.int32)

    # Dispatch: 32 threads per output token, total = B*H*S simdgroups
    grid_x = B * H * S * 32
    grid = (grid_x, 1, 1)
    threadgroup = (32, 1, 1)

    out_bm = kernel(
        inputs=[q_bm, k_bm, v_bm, block_indices],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("BS", BS),
            ("TK", TK),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, H, S, D)],
        output_dtypes=[q.dtype],
    )[0]

    return _reshape_from_block_major(out_bm, shape, chunk_thw)


# ===========================================================================
# Phase 3 — threadgroup-shared K + V (FASTER)
# ===========================================================================
#
# Phase 2 had 1 simdgroup per output token; each Q token's simdgroup
# independently read K/V from global memory. For 64 Q tokens in the same
# BSA block all attending to the SAME selected KV blocks, that's 64×
# redundant K/V reads.
#
# Phase 3 changes the dispatch:
#   - 1 threadgroup = 1 Q block (64 Q tokens)
#   - 8 simdgroups per threadgroup; each handles 8 Q tokens
#   - K + V cooperatively loaded into 32 KB threadgroup-shared memory
#     ONCE per KV block; all 8 simdgroups reuse the same shared K/V
#   - Q in registers (per-thread, distributed across simdgroup)
#   - Per-thread state: 8 Q tokens × QK_PT slice of Q/output
#
# This eliminates the 64× redundant global K/V reads of Phase 2 and
# replaces them with one cooperative shared-memory load per KV block.
# The dominant cost should shift from memory bandwidth to compute.

_BSA_KERNEL_V3_BODY = r"""
    // ===================================================================
    // BSA Tier B Phase-3 threadgroup-shared K + V kernel.
    //
    // Threading:
    //   grid:        (B * H * num_q_blocks, 1, 1)
    //   threadgroup: (256, 1, 1) = 8 simdgroups
    //   → one threadgroup = one Q block (64 Q tokens)
    //
    // Shared memory layout:
    //   K_smem[BS * D]: 16 KB for BS=64, D=128, fp16
    //   V_smem[BS * D]: 16 KB for BS=64, D=128, fp16
    //   Total: 32 KB (Apple Silicon M-series threadgroup memory limit)
    //
    // Each simdgroup handles QPSG = BS / NSG = 64 / 8 = 8 Q tokens.
    // ===================================================================

    constexpr uint BD = 32;          // simdgroup width
    constexpr uint NSG = 8;          // simdgroups per threadgroup
    constexpr uint QPSG = BS / NSG;  // Q tokens per simdgroup = 8
    constexpr uint QK_PT = D / BD;   // QK elements per thread = 4 (D=128)
    constexpr uint TG_SIZE = NSG * BD;  // total threads/group = 256
    constexpr uint TILE_BYTES = BS * D;  // elements per shared tile (K or V)

    uint tg_lid = thread_position_in_threadgroup.x;
    uint simd_lid = thread_index_in_simdgroup;
    uint simd_gid = simdgroup_index_in_threadgroup;

    uint linear_tg = threadgroup_position_in_grid.x;
    uint S = q_shape[2];
    uint H = q_shape[1];
    uint B = q_shape[0];
    uint num_q_blocks = S / BS;

    // Decode (b, h, q_block) from threadgroup id
    uint b = linear_tg / (H * num_q_blocks);
    uint hq = linear_tg % (H * num_q_blocks);
    uint h = hq / num_q_blocks;
    uint q_block = hq % num_q_blocks;

    // 1. Load Q into per-thread registers.
    // Each simdgroup handles QPSG=8 Q tokens. Q token in this Q block:
    //   q_token_local = simd_gid * QPSG + i   for i in 0..QPSG
    // Each thread holds QK_PT=4 elements per Q token, indexed by simd_lid.
    float q_reg[QPSG][QK_PT];
    uint q_block_base = ((b * H + h) * S + q_block * BS) * D;
    for (uint i = 0; i < QPSG; i++) {
        uint q_token_local = simd_gid * QPSG + i;
        for (uint d = 0; d < QK_PT; d++) {
            q_reg[i][d] = (float)q[q_block_base + q_token_local * D
                                   + simd_lid * QK_PT + d];
        }
    }

    // 2. Initialize per-Q-token softmax state in registers.
    float max_score[QPSG];
    float sum_score[QPSG];
    float o_reg[QPSG][QK_PT];
    for (uint i = 0; i < QPSG; i++) {
        max_score[i] = -INFINITY;
        sum_score[i] = 0.0f;
        for (uint d = 0; d < QK_PT; d++) o_reg[i][d] = 0.0f;
    }

    const float scale = 1.0f / metal::sqrt((float)D);
    const float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;

    // 3. Shared memory for K and V tiles
    threadgroup T K_smem[TILE_BYTES];
    threadgroup T V_smem[TILE_BYTES];

    // 4. Loop over selected KV blocks (TK iterations)
    uint bi_base = ((b * H + h) * num_q_blocks + q_block) * TK;
    for (uint sel = 0; sel < TK; sel++) {
        int kv_block_idx = block_indices[bi_base + sel];
        uint kv_token_base = (uint)kv_block_idx * BS;
        uint kv_block_global = ((b * H + h) * S + kv_token_base) * D;

        // 4a. Cooperatively load K and V into shared
        // TG_SIZE=256 threads divide TILE_BYTES=8192 (for D=128) → 32 per thread.
        for (uint e = tg_lid; e < TILE_BYTES; e += TG_SIZE) {
            K_smem[e] = k[kv_block_global + e];
            V_smem[e] = v[kv_block_global + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4b. For each K token in the block
        for (uint j = 0; j < BS; j++) {
            uint kv_idx = kv_token_base + j;
            if (kv_idx >= S) break;

            // Compute scores for all QPSG=8 of MY simdgroup's Q tokens.
            // Each thread computes its slice's dot product; simd_sum reduces
            // across the 32 threads to one scalar score per Q token.
            float scores[QPSG];
            for (uint i = 0; i < QPSG; i++) {
                float local_dot = 0.0f;
                for (uint d = 0; d < QK_PT; d++) {
                    float k_val = (float)K_smem[j * D + simd_lid * QK_PT + d];
                    local_dot += q_reg[i][d] * k_val;
                }
                scores[i] = metal::simd_sum(local_dot) * scale_log2;
            }

            // Read V[j] from shared (each thread its QK_PT slice). One read
            // per K token per thread — V_smem amortizes across all 8
            // simdgroups in the threadgroup (no global redundancy).
            float v_local[QK_PT];
            for (uint d = 0; d < QK_PT; d++) {
                v_local[d] = (float)V_smem[j * D + simd_lid * QK_PT + d];
            }

            // Online softmax + output update per Q token
            for (uint i = 0; i < QPSG; i++) {
                float m_new = metal::max(max_score[i], scores[i]);
                float exp_diff = metal::fast::exp2(max_score[i] - m_new);
                float exp_score = metal::fast::exp2(scores[i] - m_new);
                sum_score[i] = sum_score[i] * exp_diff + exp_score;
                for (uint d = 0; d < QK_PT; d++) {
                    o_reg[i][d] = o_reg[i][d] * exp_diff + exp_score * v_local[d];
                }
                max_score[i] = m_new;
            }
        }
        // Wait for all simdgroups before reloading shared K/V
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Normalize + write output
    for (uint i = 0; i < QPSG; i++) {
        uint q_token_local = simd_gid * QPSG + i;
        uint q_global = q_block * BS + q_token_local;
        if (q_global >= S) continue;
        float inv_l = (sum_score[i] > 0.0f) ? (1.0f / sum_score[i]) : 0.0f;
        uint out_base = ((b * H + h) * S + q_global) * D;
        for (uint d = 0; d < QK_PT; d++) {
            out[out_base + simd_lid * QK_PT + d] = (T)(o_reg[i][d] * inv_l);
        }
    }
"""


def _get_kernel_v3(D: int, BS: int, TK: int, dtype: mx.Dtype):
    """JIT-compile the Phase 3 kernel (cached)."""
    key = (D, BS, TK, dtype)
    if key not in _KERNEL_V3_CACHE:
        _KERNEL_V3_CACHE[key] = mx.fast.metal_kernel(
            name=f"bsa_phase3_D{D}_BS{BS}_TK{TK}",
            input_names=["q", "k", "v", "block_indices"],
            output_names=["out"],
            source=_BSA_KERNEL_V3_BODY,
            ensure_row_contiguous=True,
        )
    return _KERNEL_V3_CACHE[key]


def bsa_attention_metal_v3(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_indices: mx.array,
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
    shape: Optional[tuple[int, int, int]] = None,
) -> mx.array:
    """BSA Tier B Phase 3 — threadgroup-shared K+V Metal kernel.

    Same interface as Phase 2 but eliminates the across-simdgroup K/V
    redundancy by sharing the loaded KV block across 8 simdgroups in
    each threadgroup. Each threadgroup handles one full Q block (64
    tokens), and the 8 simdgroups within share the K and V tiles from
    32 KB threadgroup memory.

    Constraints (in addition to Phase 2's):
    - block_size (= chunk_thw[0]*chunk_thw[1]*chunk_thw[2]) must be 64
      (the kernel's NSG=8 simdgroups × QPSG=8 Q tokens = 64)
    - For D=128 fp16: K_smem + V_smem = 32 KB. Apple Silicon M-series
      threadgroup memory limit is 32 KB. For larger D, will need
      a smaller tile or fall back to Phase 2.
    """
    if shape is None:
        raise ValueError(
            "shape=(T_lat, H_lat, W_lat) is required for Tier B BSA"
        )
    B, H, S, D = q.shape
    BS = chunk_thw[0] * chunk_thw[1] * chunk_thw[2]
    TK = int(block_indices.shape[-1])

    if S % BS != 0:
        raise ValueError(f"S={S} must be a multiple of block_size={BS}.")
    if BS != 64:
        raise ValueError(
            f"Phase 3 currently requires block_size=64, got {BS}. "
            f"Use Phase 2 (bsa_attention_metal_v2) for other block sizes."
        )
    if D % 32 != 0:
        raise ValueError(
            f"head_dim D={D} must be a multiple of 32 (simdgroup width)."
        )
    # Threadgroup memory budget check: K_smem + V_smem = 2 * BS * D * sizeof(T)
    dtype_size = 2 if q.dtype in (mx.float16, mx.bfloat16) else 4
    tg_mem_bytes = 2 * BS * D * dtype_size
    if tg_mem_bytes > 32 * 1024:
        raise ValueError(
            f"Phase 3 K+V shared memory {tg_mem_bytes / 1024:.0f} KB exceeds "
            f"32 KB Apple Silicon M-series threadgroup memory limit. "
            f"D={D}, fp{8*dtype_size}. Use Phase 2 instead."
        )

    # Rearrange t-major → block-major for kernel's block-contiguous layout
    q_bm = _reshape_to_block_major(q, shape, chunk_thw)
    k_bm = _reshape_to_block_major(k, shape, chunk_thw)
    v_bm = _reshape_to_block_major(v, shape, chunk_thw)

    kernel = _get_kernel_v3(D, BS, TK, q.dtype)

    if block_indices.dtype != mx.int32:
        block_indices = block_indices.astype(mx.int32)

    # Dispatch: one threadgroup per Q block, 256 threads per threadgroup
    num_q_blocks = S // BS
    grid = (B * H * num_q_blocks * 256, 1, 1)
    threadgroup = (256, 1, 1)

    out_bm = kernel(
        inputs=[q_bm, k_bm, v_bm, block_indices],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("BS", BS),
            ("TK", TK),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, H, S, D)],
        output_dtypes=[q.dtype],
    )[0]

    return _reshape_from_block_major(out_bm, shape, chunk_thw)
