"""PT↔MLX parity for a single LongCatSingleStreamBlock.

This is the **structural drift detector** (mlx-porting skill pitfall #9):
forward a small randomly-initialized DiT block through both PT and MLX,
verify the outputs match within fp32 numerical tolerance. Catches:

- AdaLN-Zero modulation arithmetic bugs (the L11/L42 family — adaLN
  expecting fp32, scale/shift unpacking order, gate residual)
- QKV interleaving / fused-QKV reshape mistakes
- Cross-attention concatenation order (visual + text packing)
- RoPE 3D index math drift
- FFN activation choice (SwiGLU vs GELU mismatch)
- Layer norm epsilon mismatches

Uses a **tiny config** (hidden=64, depth=1, num_heads=4) so the
random-weight forward runs in <1s on CPU. Weights are seeded and
shared between PT and MLX — no checkpoint download needed.

The threshold is 5e-3 max_abs (per the mlx-porting skill: "full
transformer block: < 5e-3") in fp32. We use CPU stream for MLX to
avoid the ~3.8e-3-per-matmul tf32-like drift that Metal-GPU fp32 has
(L11 — relevant numerics lesson from the Zonos port too).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:
    import torch
    _PT_AVAILABLE = True
except Exception:
    _PT_AVAILABLE = False

import mlx.core as mx

# Import our MLX block at module load time — BEFORE the fixture swaps
# `sys.modules["longcat_video"]` for upstream's virtual package.
from longcat_video.models.longcat_video_dit import (
    LongCatSingleStreamBlock as MXBlock,
)
from tests.parity._helpers import assert_parity

REFS = pathlib.Path(__file__).resolve().parents[2] / "refs" / "longcat-video"


@pytest.fixture(scope="module")
def upstream_pt_block():
    """Build a PT LongCatSingleStreamBlock from the upstream source.

    The upstream code reaches beyond `modules/` for distributed-training
    helpers (`context_parallel`, `block_sparse_attention`) that have no
    effect on a single-GPU forward. We stub them with no-op modules so
    the import resolves cleanly. Skipped if `refs/longcat-video/` is
    missing OR if a different sub-import we can't stub blocks the load.
    """
    if not REFS.exists():
        pytest.skip(
            f"refs/longcat-video not present (expected at {REFS}). "
            "Symlink or clone the upstream repo to enable this test."
        )

    import importlib.util
    import types

    # Snapshot any sys.modules entries our installed `longcat_video` left
    # behind. We need to swap them out for upstream's virtual package
    # during the test, then restore on teardown.
    saved_modules = {
        k: v for k, v in sys.modules.items()
        if k == "longcat_video" or k.startswith("longcat_video.")
    }
    for k in list(saved_modules):
        del sys.modules[k]

    # Build a virtual `longcat_video` package rooted at refs/
    lc_root = REFS / "longcat_video"
    if not lc_root.exists():
        pytest.skip(f"upstream longcat_video/ not found at {lc_root}")
    lc_pkg = types.ModuleType("longcat_video")
    lc_pkg.__path__ = [str(lc_root)]
    sys.modules["longcat_video"] = lc_pkg

    # context_parallel stub — no-op functions used by upstream's attention.
    # Mark as a package so submodule lookups (`.ulysses_wrapper`) resolve.
    cp = types.ModuleType("longcat_video.context_parallel")
    cp.__path__ = []  # make it a "package"

    def _identity(*a, **kw): return a[0] if a else None

    cp_util = types.ModuleType("longcat_video.context_parallel.context_parallel_util")
    cp_util.get_cp_rank = lambda: 0
    cp_util.get_cp_size = lambda: 1
    cp_util.is_cp_initialized = lambda: False
    cp_util.context_parallel_split = _identity
    cp_util.context_parallel_gather = _identity
    cp_util.is_context_parallel_initialized = lambda: False

    ulysses = types.ModuleType("longcat_video.context_parallel.ulysses_wrapper")
    ulysses.ulysses_wrapper = _identity   # attention.py imports this exact symbol
    ulysses.ulysses_attention = _identity
    ulysses.UlyssesWrapper = type("UlyssesWrapper", (), {})

    cp.context_parallel_util = cp_util
    cp.ulysses_wrapper = ulysses
    sys.modules["longcat_video.context_parallel"] = cp
    sys.modules["longcat_video.context_parallel.context_parallel_util"] = cp_util
    sys.modules["longcat_video.context_parallel.ulysses_wrapper"] = ulysses

    # block_sparse_attention stub — single-GPU forward doesn't use it
    bsa = types.ModuleType("longcat_video.block_sparse_attention")
    bsa_interface = types.ModuleType(
        "longcat_video.block_sparse_attention.bsa_interface"
    )
    bsa_interface.flash_attn_bsa_3d = _identity
    bsa_interface.flash_attn_bsa_3d_varlen = _identity
    bsa_interface.flash_attn_bsa_varlen_kvpacked = _identity
    bsa.bsa_interface = bsa_interface
    sys.modules["longcat_video.block_sparse_attention"] = bsa
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = bsa_interface

    # Now load the DiT submodule via importlib
    modules_dir = REFS / "longcat_video" / "modules"
    dit_spec = importlib.util.spec_from_file_location(
        "longcat_video.modules.longcat_video_dit",
        modules_dir / "longcat_video_dit.py",
    )
    if dit_spec is None:
        pytest.skip("could not build spec for upstream DiT")
    dit_mod = importlib.util.module_from_spec(dit_spec)
    sys.modules["longcat_video.modules.longcat_video_dit"] = dit_mod
    try:
        dit_spec.loader.exec_module(dit_mod)
    except Exception as e:
        pytest.skip(f"upstream longcat_video_dit.py failed to load: {e}")

    # Yield the class; restore sys.modules so subsequent tests in the
    # same pytest session don't pick up upstream stubs.
    PTBlock = dit_mod.LongCatSingleStreamBlock
    yield PTBlock
    # --- teardown ---
    for k in list(sys.modules):
        if k == "longcat_video" or k.startswith("longcat_video."):
            del sys.modules[k]
    sys.modules.update(saved_modules)


@pytest.mark.skipif(
    not _PT_AVAILABLE,
    reason="parity dep missing — install with `pip install -e '.[parity]'`",
)
def test_dit_block_forward_parity_cpu(upstream_pt_block):
    """Run one DiT block PT vs MLX (CPU stream) and assert max_abs < 5e-3."""
    # Tiny config — keeps the random-init forward fast
    hidden_size = 64
    num_heads = 4
    mlp_ratio = 2
    adaln_tembed_dim = 32

    # --- PT side: build, seed-init, eval mode ---
    # `cp_split_hw=[1, 1]` — upstream's RoPE module unconditionally
    # subscripts `cp_split_hw[0]` even when there's no context_parallel
    # (latent upstream bug; they always run multi-GPU).
    torch.manual_seed(0)
    pt_block = upstream_pt_block(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        adaln_tembed_dim=adaln_tembed_dim,
        cp_split_hw=[1, 1],
    )
    pt_block.eval()

    # Monkey-patch upstream's `_process_attn` to use PT's standard SDPA on
    # CPU. Upstream requires flash-attn / xformers / BSA, none of which
    # work on CPU. The SDPA path is the dense-equivalent of every backend
    # (the BSA backend is a sparse approximation thereof).
    def _sdpa_attn(self, q, k, v, shape):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale,
        )

    pt_block.attn._process_attn = _sdpa_attn.__get__(pt_block.attn)
    # Cross-attention's path is INLINE in its forward (not a separate
    # _process_attn method), so we replace the whole forward.
    def _cross_attn_forward_sdpa(self, x, cond, kv_seqlen,
                                  num_cond_latents=None, shape=None):
        B = x.shape[0]
        N = x.shape[1]
        C = x.shape[2]
        scale = self.head_dim ** -0.5
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        # Standard SDPA — single-batch text (B=1 packed) → broadcast to B
        if k.shape[0] != q.shape[0]:
            k = k.expand(q.shape[0], -1, -1, -1)
            v = v.expand(q.shape[0], -1, -1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)

    pt_block.cross_attn.forward = _cross_attn_forward_sdpa.__get__(pt_block.cross_attn)

    # --- MLX side: build with default init, then COPY weights from PT ---
    mx_block = MXBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        adaln_tembed_dim=adaln_tembed_dim,
    )

    # Copy PT params → MLX (Linear weight layout is identical; no Conv*d here)
    from mlx.utils import tree_flatten, tree_unflatten

    pt_sd = pt_block.state_dict()
    mx_keys = {k for k, _ in tree_flatten(mx_block.parameters())}

    flat = []
    unmapped_pt = []
    for pt_k, pt_v in pt_sd.items():
        # PT keys may differ slightly from MX keys — adapt the common cases.
        # Our blocks.py mirrors upstream 1:1 so direct map should work.
        if pt_k in mx_keys:
            arr = pt_v.detach().cpu().float().numpy()
            flat.append((pt_k, mx.array(arr)))
        else:
            unmapped_pt.append(pt_k)

    # Some keys may be init-only (e.g. norm-affine off) — those won't appear
    # in either tree. Sanity-bound unmapped.
    assert len(unmapped_pt) < 5, (
        f"Too many unmapped PT keys ({len(unmapped_pt)}): {unmapped_pt[:5]}"
    )
    mx_block.update(tree_unflatten(flat))
    mx.eval(mx_block.parameters())

    # --- Seeded inputs (identical for both) ---
    B, T, H_lat, W_lat = 1, 2, 2, 2
    N = T * H_lat * W_lat  # 8 tokens for the visual stream
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, N, hidden_size)).astype(np.float32) * 0.1
    y_np = rng.standard_normal((1, 4, hidden_size)).astype(np.float32) * 0.1
    t_np = rng.standard_normal((B, T, adaln_tembed_dim)).astype(np.float32) * 0.5
    y_seqlen = [4]
    latent_shape = (T, H_lat, W_lat)

    # --- PT forward ---
    with torch.no_grad():
        pt_out = pt_block(
            torch.from_numpy(x_np),
            torch.from_numpy(y_np),
            torch.from_numpy(t_np),
            y_seqlen,
            latent_shape,
        )
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]

    # --- MLX forward — pin to CPU stream for fp32 stability (L11) ---
    with mx.stream(mx.cpu):
        mx_out = mx_block(
            mx.array(x_np),
            mx.array(y_np),
            mx.array(t_np),
            y_seqlen,
            latent_shape,
        )
        mx.eval(mx_out)

    assert_parity(pt_out, mx_out, threshold=5e-3, name="dit_block.cpu")
