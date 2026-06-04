"""Smoke tests for recipes/convert_longcat_video.py — no network, no weights.

Verifies the conversion logic against tiny synthetic dicts: layout transposes,
dtype casts, materialization, sharding behavior, fp32-stays-fp32 predicate.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import numpy as np

from recipes.convert_longcat_video import (
    _key_should_stay_fp32,
    _layout_and_cast,
    _materialize_and_save,
    _save_sharded_safetensors,
)


def test_layout_conv1d_transposes():
    """Conv1d (O, I, K) → (O, K, I)."""
    arr = mx.zeros((8, 4, 3), dtype=mx.float32)
    out = _layout_and_cast(arr, name="conv1.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 3, 4)
    assert out.dtype == mx.bfloat16


def test_layout_conv3d_transposes():
    """Conv3d (O, I, T, H, W) → (O, T, H, W, I)."""
    arr = mx.zeros((8, 4, 3, 3, 3), dtype=mx.float32)
    out = _layout_and_cast(arr, name="x_embedder.proj.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 3, 3, 3, 4)


def test_layout_2d_passes_through():
    """Linear weights / 2D arrays are not transposed."""
    arr = mx.zeros((8, 4), dtype=mx.float32)
    out = _layout_and_cast(arr, name="blocks.0.attn.qkv.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 4)


def test_layout_gamma_no_transpose():
    """RMS gamma keeps its rank even if 4D/5D-shaped."""
    arr = mx.ones((384, 1, 1, 1), dtype=mx.float32)
    out = _layout_and_cast(arr, name="norm.gamma", is_gamma=True, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (384, 1, 1, 1)


def test_adaLN_modulation_keeps_fp32():
    """Per CLAUDE.md L11: adaLN_modulation Linears must stay fp32."""
    arr = mx.zeros((8, 4), dtype=mx.float32)
    out = _layout_and_cast(
        arr, name="blocks.0.adaLN_modulation.1.weight", is_gamma=False, dtype=mx.bfloat16
    )
    mx.eval(out)
    assert out.dtype == mx.float32, "adaLN_modulation must stay fp32 (silent accumulation bug otherwise)"


def test_adaLN_modulation_upcasts_when_source_bf16():
    """A bf16 source for an adaLN key must be force-upcast to fp32."""
    bf16_arr = mx.zeros((8, 4)).astype(mx.bfloat16)
    out = _layout_and_cast(
        bf16_arr, name="blocks.0.adaLN_modulation.1.weight", is_gamma=False, dtype=mx.bfloat16
    )
    mx.eval(out)
    assert out.dtype == mx.float32


def test_key_should_stay_fp32_predicate():
    """The fp32-stay predicate gates on `adaLN_modulation` substring."""
    assert _key_should_stay_fp32("blocks.0.adaLN_modulation.1.weight")
    assert _key_should_stay_fp32("final_layer.adaLN_modulation.1.weight")
    assert not _key_should_stay_fp32("blocks.0.attn.qkv.weight")
    assert not _key_should_stay_fp32("blocks.0.norm1.weight")


def test_materialize_writes_nonzero(tmp_path):
    """Lazy MLX tensors must be evaluated before serialization."""
    a = mx.ones((4, 4))
    b = mx.random.normal((8, 8))
    c = a @ b[:4, :4]  # forces a lazy matmul
    sd = {"a": a, "b": b, "c": c}
    out_path = tmp_path / "test.safetensors"
    _materialize_and_save(sd, out_path)

    from safetensors import safe_open
    with safe_open(str(out_path), framework="numpy") as f:
        for k in ("a", "b", "c"):
            arr = f.get_tensor(k)
            assert np.abs(arr).sum() > 0, f"{k} was saved as zeros — silent killer"


def test_sharding_writes_index_and_multiple_shards(tmp_path):
    """Sharding produces N safetensors + an index.json mapping every key."""
    sd = {f"k{i}": mx.random.normal((256, 256)) for i in range(8)}
    _save_sharded_safetensors(sd, tmp_path / "weights", base_name="model",
                              max_shard_size_bytes=1 * 1024 * 1024)
    shards = sorted((tmp_path / "weights").glob("model-*.safetensors"))
    assert len(shards) > 1
    idx = json.loads((tmp_path / "weights" / "model.safetensors.index.json").read_text())
    assert len(idx["weight_map"]) == 8


def test_recipe_module_imports():
    """The recipe imports + has the orchestrator entrypoint."""
    from recipes.convert_longcat_video import (
        build_bf16_variant,
        convert_dit,
        convert_lora,
        convert_umt5,
        convert_vae,
        BASE_REPO,
        PUBLISH_REPO_BF16,
    )
    assert BASE_REPO == "meituan-longcat/LongCat-Video"
    assert PUBLISH_REPO_BF16 == "mlx-community/LongCat-Video-bf16"
    assert callable(build_bf16_variant)
