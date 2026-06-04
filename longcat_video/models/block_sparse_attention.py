"""Block Sparse Attention (BSA) — **Tier A: pure-MLX reference implementation**.

This is the *correctness* implementation: same routing decisions and same
final attention output as upstream's Triton kernel, but done with vanilla
MLX ops + dense `scaled_dot_product_attention` with a sparsity-induced
additive mask. Tier B (Metal kernel, B4.1) is what will make this fast
by actually skipping the non-routed K/V blocks at the kernel level.

## Why Tier A is correct-but-not-fast

The token-pair attention mask we build is exactly `True` on
`(s_q, s_k)` pairs whose 3-D blocks were routed together by the top-k
gate, and `False` (→ -inf in the additive mask) elsewhere. So the SDPA
output is mathematically identical to what a perfect BSA kernel would
emit. But because we still materialize the full `[B, H, S, S]` mask and
run a dense matmul, we get no FLOPs win — just correctness.

## Algorithm (upstream `bsa_interface.py`)

1. **Mean-pool compress** Q and K into per-block representatives:
   `[B, H, S, D]` → `[B, H, num_blocks, D]` where `num_blocks = (T_lat * H_lat * W_lat) / 64`
   and tokens are grouped into `[chunk_t=4, chunk_h=4, chunk_w=4]` voxels.
2. **Block-level routing**: `score = q_blocks @ k_blocks^T`
   → `[B, H, num_q_blocks, num_kv_blocks]`
3. **Top-k**: per Q-block, keep `(1 - sparsity)` fraction of K/V blocks
   (config: sparsity=0.9375 → keep 6.25% = `num_selected` of `num_kv_blocks`)
4. **Expand** the block-pair routing to a token-pair additive mask via
   `block_of_token[s] = bt * Hb * Wb + bh * Wb + bw` for each token's
   `(t, h, w)` position in the latent grid.
5. **Dense SDPA** with that mask — output is identical to a sparse kernel
   that only computed the routed pairs.

## Constraints we rely on

- `T_lat`, `H_lat`, `W_lat` are all multiples of 4 — guaranteed by the
  refinement pipeline's BSA-alignment padding (`bsa_latent_granularity=4`
  in `refinement.py`).
- Token ordering on the sequence axis matches the canonical
  `(t, h, w)` t-major flatten — same as everywhere else in the DiT.

PT reference: `refs/longcat-video/longcat_video/block_sparse_attention/bsa_interface.py`
- `mean_pooling_compression` (line 169)
- `cal_score` (line 181)
- `get_select_indices_topk_from_score` (line 213)
- `attn_fwd_bsa_varlen_triton` (line 290) — the Tier B kernel we replace
  with dense SDPA + mask here.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


__all__ = [
    "mean_pool_blocks_3d",
    "block_routing_scores",
    "topk_block_indices",
    "build_token_block_indices",
    "build_token_pair_mask",
    "bsa_attention",
]


def mean_pool_blocks_3d(
    x: mx.array,
    shape: tuple[int, int, int],
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
) -> mx.array:
    """Mean-pool a per-token feature tensor into per-block representatives.

    Args:
        x: `[B, H, S, D]` where `S = T_lat * H_lat * W_lat` in (t, h, w) major.
        shape: (T_lat, H_lat, W_lat) — must all be multiples of chunk_thw.
        chunk_thw: block voxel shape (default (4, 4, 4) per BSA config).

    Returns:
        `[B, H, num_blocks, D]` mean-pooled block representatives, ordered
        in block-major `(bt, bh, bw)` flatten (compatible with downstream
        `bt * Hb * Wb + bh * Wb + bw` indexing).
    """
    T, H_, W = shape
    cT, cH, cW = chunk_thw
    if T % cT or H_ % cH or W % cW:
        raise ValueError(
            f"Latent shape ({T}, {H_}, {W}) must be a multiple of "
            f"chunk_thw {chunk_thw}; refinement padding should guarantee this."
        )
    Bt, Bh, Bw = T // cT, H_ // cH, W // cW
    B, H, S, D = x.shape
    if S != T * H_ * W:
        raise ValueError(
            f"S={S} doesn't match T*H*W={T*H_*W} for shape {shape}"
        )

    # [B, H, T, H_, W, D] → group block-internal axes → mean over them
    y = x.reshape(B, H, T, H_, W, D)
    y = y.reshape(B, H, Bt, cT, Bh, cH, Bw, cW, D)
    # Permute to (B, H, Bt, Bh, Bw, cT, cH, cW, D) then mean over (cT,cH,cW)
    y = y.transpose(0, 1, 2, 4, 6, 3, 5, 7, 8)
    y = y.reshape(B, H, Bt * Bh * Bw, cT * cH * cW, D)
    return y.mean(axis=3)


def block_routing_scores(q_blocks: mx.array, k_blocks: mx.array) -> mx.array:
    """`q_blocks, k_blocks`: `[B, H, num_blocks, D]`.

    Returns `[B, H, num_q_blocks, num_kv_blocks]` raw routing scores.
    """
    return mx.matmul(q_blocks, k_blocks.swapaxes(-1, -2))


def topk_block_indices(score: mx.array, sparsity: float) -> tuple[mx.array, int]:
    """Per Q-block top-k routing.

    Args:
        score: `[B, H, num_q_blocks, num_kv_blocks]`.
        sparsity: fraction of KV blocks to *drop* (0.9375 keeps 6.25%).

    Returns:
        - block_indices: `[B, H, num_q_blocks, num_selected]` int32
        - num_selected: int (always at least 1)
    """
    num_kv_blocks = int(score.shape[-1])
    num_selected = max(1, int((1.0 - sparsity) * num_kv_blocks))
    # MLX argpartition is bottom-k; flip sign for top-k by score
    # Then take the first num_selected
    neg_score = -score
    # argpartition signature: kth on selected axis; we want kth so the first
    # num_selected positions (lowest of neg_score = highest of score) are kth-best
    part = mx.argpartition(neg_score, kth=num_selected - 1, axis=-1)
    return part[..., :num_selected].astype(mx.int32), num_selected


def build_token_block_indices(
    shape: tuple[int, int, int],
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
) -> mx.array:
    """For each token at flat index `s` (in t-major `(t, h, w)` ordering),
    return the index of the BSA block it belongs to (in `(bt, bh, bw)`
    block-major ordering).

    Returns: `[S]` int32, where S = T_lat * H_lat * W_lat.
    """
    T, H_, W = shape
    cT, cH, cW = chunk_thw
    Bh, Bw = H_ // cH, W // cW

    s = mx.arange(T * H_ * W, dtype=mx.int32)
    t = s // (H_ * W)
    hw = s % (H_ * W)
    h = hw // W
    w = hw % W
    bt = t // cT
    bh = h // cH
    bw = w // cW
    return bt * (Bh * Bw) + bh * Bw + bw


def build_token_pair_mask(
    block_indices: mx.array,
    token_block_index: mx.array,
    num_kv_blocks: int,
) -> mx.array:
    """Expand a per-block routing decision into a per-token pair mask.

    Args:
        block_indices: `[B, H, num_q_blocks, num_selected]` int32, output of
                       topk_block_indices.
        token_block_index: `[S]` int32, the block each token belongs to
                           (output of build_token_block_indices).
        num_kv_blocks: total number of KV blocks (= num_q_blocks here).

    Returns:
        `[B, H, S, S]` bool — `True` if `(s_q, s_k)`'s blocks were routed
        together by the top-k gate.

    The materialization is full-S² so this is *correct but not fast*.
    Tier B (Metal kernel) skips this and reads block_indices directly.
    """
    B, H, num_q_blocks, num_selected = block_indices.shape

    # Step 1: one_hot → [B, H, num_q_blocks, num_selected, num_kv_blocks] bool
    # then collapse num_selected with any() → [B, H, num_q_blocks, num_kv_blocks]
    k_range = mx.arange(num_kv_blocks, dtype=mx.int32)
    one_hot = (block_indices[..., None] == k_range[None, None, None, None, :])
    block_pair_mask = mx.any(one_hot, axis=3)  # [B, H, num_q_blocks, num_kv_blocks]

    # Step 2: token-pair mask via advanced indexing along last two axes.
    # mask[..., s_q, s_k] = block_pair_mask[..., block_of(s_q), block_of(s_k)]
    # We use mx.take along axis=-2 then axis=-1 for shape clarity.
    # block_pair_mask[..., token_block_index, :]: [B, H, S, num_kv_blocks]
    expanded_q = mx.take(block_pair_mask, token_block_index, axis=-2)
    # then expand on the K side
    token_pair_mask = mx.take(expanded_q, token_block_index, axis=-1)
    return token_pair_mask


def bsa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    shape: tuple[int, int, int],
    sparsity: float = 0.9375,
    chunk_thw: tuple[int, int, int] = (4, 4, 4),
    sm_scale: Optional[float] = None,
) -> mx.array:
    """Pure-MLX Block Sparse Attention (Tier A).

    Args:
        q, k, v: `[B, H, S, D]` where `S = T_lat * H_lat * W_lat`
        shape: (T_lat, H_lat, W_lat) — must all be multiples of chunk_thw
        sparsity: fraction of KV blocks dropped per Q block (default 0.9375)
        chunk_thw: BSA block voxel shape (default (4, 4, 4))
        sm_scale: softmax scale; defaults to `head_dim ** -0.5`

    Returns: `[B, H, S, D]` — mathematically identical to a true BSA kernel.
    """
    if sm_scale is None:
        D = int(q.shape[-1])
        sm_scale = 1.0 / math.sqrt(D)

    # 1. Compress
    q_blocks = mean_pool_blocks_3d(q, shape, chunk_thw)
    k_blocks = mean_pool_blocks_3d(k, shape, chunk_thw)

    # 2. Route
    score = block_routing_scores(q_blocks, k_blocks)

    # 3. Top-k per Q block
    block_indices, num_selected = topk_block_indices(score, sparsity)

    # 4. Expand to token-pair mask
    token_block_index = build_token_block_indices(shape, chunk_thw)
    num_kv_blocks = int(score.shape[-1])
    token_pair_mask = build_token_pair_mask(
        block_indices, token_block_index, num_kv_blocks,
    )

    # 5. Dense SDPA with additive mask
    additive_mask = mx.where(
        token_pair_mask,
        mx.zeros((), dtype=q.dtype),
        mx.full((), -1e9, dtype=q.dtype),
    )
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=sm_scale, mask=additive_mask,
    )
