"""Smoke tests for Block Sparse Attention Tier B Metal kernel.

Mirrors `test_bsa_tier_a_smoke.py::test_bsa_sparsity_zero_matches_dense`
— the L24 degenerate-case correctness gate — but for the Metal kernel.

Threshold rationale: Metal GPU fp32 matmul is tf32-like (~3.8e-3 per
matmul per L11), so the tolerance is relaxed vs CPU. Our kernel uses
fp32 accumulators on fp16 inputs, which gives a tight bound.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest


def _import_smoke():
    from longcat_video.models.block_sparse_attention import (
        bsa_attention,
        topk_block_indices,
        block_routing_scores,
        mean_pool_blocks_3d,
    )
    from longcat_video.models.block_sparse_attention_metal import (
        bsa_attention_metal,
        prewarm_metal_kernel,
    )
    return (
        bsa_attention, bsa_attention_metal,
        topk_block_indices, block_routing_scores, mean_pool_blocks_3d,
        prewarm_metal_kernel,
    )


def test_imports():
    funcs = _import_smoke()
    assert all(f is not None for f in funcs)


def test_metal_kernel_degenerate_case_matches_dense_fp32():
    """L24 again — the canonical correctness gate. With one Q block + one
    KV block, top_k=1 means every Q block attends to that one KV block,
    which IS the full sequence. So the kernel reduces to dense SDPA.
    """
    _, bsa_metal, *_ = _import_smoke()

    # 4×4×4 latent = 1 block of 64 tokens. Single Q & KV block.
    T, H_lat, W_lat = 4, 4, 4
    S = T * H_lat * W_lat   # = 64
    B, num_heads, D = 1, 2, 16

    mx.random.seed(0)
    q = mx.random.normal((B, num_heads, S, D))
    k = mx.random.normal((B, num_heads, S, D))
    v = mx.random.normal((B, num_heads, S, D))
    mx.eval(q, k, v)

    sm_scale = 1.0 / math.sqrt(D)
    out_dense = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=sm_scale,
    )

    # 1 Q block × 1 KV block → block_indices = [[[[0]]]] for each (b, h)
    block_indices = mx.zeros((B, num_heads, 1, 1), dtype=mx.int32)
    out_metal = bsa_metal(q, k, v, block_indices, chunk_thw=(4, 4, 4), shape=(T, H_lat, W_lat))
    mx.eval(out_dense, out_metal)

    diff = float(mx.max(mx.abs(out_metal - out_dense)))
    # Metal-GPU fp32 tolerance — see L11 + bsa-tier-a-design.md
    assert diff < 5e-3, (
        f"Tier B degenerate case (sparsity=0 ≡ dense): max_abs={diff:.3e}"
    )


def test_metal_kernel_matches_tier_a_dense_fallback():
    """Tier B with `block_indices = arange(num_blocks)` (no sparsity)
    must match Tier A with sparsity=0, which itself matches dense.

    Uses a 2-block scenario to verify the multi-block loop logic.
    """
    bsa_a, bsa_metal, *_ = _import_smoke()

    # 8×4×4 latent → T_lat=8, so we have 2 BSA blocks along T axis
    T, H_lat, W_lat = 8, 4, 4
    S = T * H_lat * W_lat   # = 128 = 2 blocks of 64
    B, num_heads, D = 1, 2, 16
    num_blocks = 2

    mx.random.seed(7)
    q = mx.random.normal((B, num_heads, S, D))
    k_arr = mx.random.normal((B, num_heads, S, D))
    v = mx.random.normal((B, num_heads, S, D))
    mx.eval(q, k_arr, v)

    # Tier A baseline
    out_a = bsa_a(q, k_arr, v, shape=(T, H_lat, W_lat), sparsity=0.0)

    # Tier B: every Q block attends to ALL KV blocks (no sparsity)
    # block_indices: [B, H, num_q_blocks=2, top_k=2]
    bi = mx.broadcast_to(
        mx.arange(num_blocks, dtype=mx.int32),
        (B, num_heads, num_blocks, num_blocks),
    )
    out_b = bsa_metal(q, k_arr, v, bi, chunk_thw=(4, 4, 4), shape=(T, H_lat, W_lat))
    mx.eval(out_a, out_b)

    diff = float(mx.max(mx.abs(out_b - out_a)))
    assert diff < 5e-3, (
        f"Tier B vs Tier A (no sparsity): max_abs={diff:.3e}"
    )


def test_metal_kernel_partial_sparsity_matches_tier_a():
    """At an intermediate sparsity (top_k = num_kv_blocks / 2), Tier B's
    kernel output must match Tier A's pure-MLX path within fp32 noise.
    Both consume the SAME `block_indices`, so the only difference is the
    computational path (kernel vs masked dense SDPA)."""
    bsa_a, bsa_metal, topk, scores_fn, mean_pool, _ = _import_smoke()

    # 8×8×8 latent = 2×2×2 = 8 blocks of 64 tokens
    T, H_lat, W_lat = 8, 8, 8
    S = T * H_lat * W_lat   # = 512
    B, num_heads, D = 1, 4, 32
    num_blocks = (T // 4) * (H_lat // 4) * (W_lat // 4)   # = 8

    mx.random.seed(42)
    q = mx.random.normal((B, num_heads, S, D))
    k_arr = mx.random.normal((B, num_heads, S, D))
    v = mx.random.normal((B, num_heads, S, D))
    mx.eval(q, k_arr, v)

    # Compute routing decisions ONCE — both Tier A and Tier B consume them
    q_blocks = mean_pool(q, shape=(T, H_lat, W_lat))
    k_blocks = mean_pool(k_arr, shape=(T, H_lat, W_lat))
    score = scores_fn(q_blocks, k_blocks)
    block_indices, n_selected = topk(score, sparsity=0.5)   # keep 4/8
    assert n_selected == 4

    # Tier A path with the same routing
    # (bsa_attention does its own routing internally — to match, we'd
    # need a "use these indices" entry point. For now, just call both
    # with sparsity=0.5 and trust that the same seeded inputs give the
    # same routing decisions.)
    out_a = bsa_a(q, k_arr, v, shape=(T, H_lat, W_lat), sparsity=0.5)

    # Tier B path
    out_b = bsa_metal(q, k_arr, v, block_indices, chunk_thw=(4, 4, 4), shape=(T, H_lat, W_lat))
    mx.eval(out_a, out_b)

    diff = float(mx.max(mx.abs(out_b - out_a)))
    assert diff < 5e-3, (
        f"Tier B vs Tier A (sparsity=0.5): max_abs={diff:.3e}"
    )


def test_metal_kernel_rejects_misaligned_S():
    """If S isn't a multiple of block_size, Tier B raises. The refinement
    pipeline pads to multiples of 4 in each spatial dim so this never
    triggers in production."""
    _, bsa_metal, *_ = _import_smoke()

    # 3×4×4 latent → S=48, not a multiple of 64
    q = mx.zeros((1, 1, 48, 16))
    k = mx.zeros((1, 1, 48, 16))
    v = mx.zeros((1, 1, 48, 16))
    bi = mx.zeros((1, 1, 1, 1), dtype=mx.int32)
    with pytest.raises(ValueError, match="multiple of"):
        bsa_metal(q, k, v, bi, chunk_thw=(4, 4, 4), shape=(4, 4, 4))


def test_prewarm_runs_without_error():
    """Prewarm should JIT-compile cached kernels without crashing."""
    *_, prewarm = _import_smoke()
    prewarm(head_dim=16, block_size=64, top_k_values=(1,), dtype=mx.float32)
