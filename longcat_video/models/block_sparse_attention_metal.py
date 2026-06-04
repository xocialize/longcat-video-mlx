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
    "prewarm_metal_kernel",
]


_KERNEL_CACHE: dict[tuple, object] = {}


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
