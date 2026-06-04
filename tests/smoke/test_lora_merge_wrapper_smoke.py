"""Smoke tests for the `scripts/_common.merge_lora` convenience wrapper.

The underlying `merge_lora_into_model` is already covered by
`test_lora_smoke.py`. This module focuses on the wrapper's behavior:

1. FileNotFoundError when the LoRA file is missing (with a helpful message
   pointing at the conversion recipe)
2. AssertionError with a clear diagnostic when 0 modules merged (catches
   the silent-no-op-merge footgun)
3. Successful merge propagates the delta into the model's parameters
   and frees the LoRA state dict
"""

from __future__ import annotations

import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest
from safetensors.numpy import save_file

# Make `scripts/_common.py` importable as a module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))


def _tiny_dit_with_one_target():
    """Build a tiny stand-in for the DiT with the exact parameter naming
    pattern the LoRA targets use: `blocks.0.attn.qkv.weight`.
    """

    class TinyDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [self._BlockStub()]

        class _BlockStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = TinyDit._AttnStub()

        class _AttnStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(8, 24, bias=False)

    return TinyDit()


def _make_lora_state_dict_for(module_path: str, in_dim: int, out_dim: int, rank: int = 4):
    """Build an encoded LoRA safetensors-style dict targeting `module_path`."""
    # PT key encoding (Meituan): `lora___lorahyphen___<path with . → ___lorahyphen___>.<tail>`
    encoded = module_path.replace(".", "___lorahyphen___")
    prefix = f"lora___lorahyphen___{encoded}"
    import numpy as np
    return {
        f"{prefix}.lora_down.weight": np.random.randn(rank, in_dim).astype("float32") * 0.01,
        f"{prefix}.lora_up.weight": np.random.randn(out_dim, rank).astype("float32") * 0.01,
    }


def test_merge_lora_file_not_found(tmp_path: pathlib.Path):
    from _common import merge_lora
    dit = _tiny_dit_with_one_target()
    variant_dir = tmp_path / "LongCat-Video-bf16"
    (variant_dir / "lora").mkdir(parents=True)
    # NOTE: lora file intentionally not created
    with pytest.raises(FileNotFoundError, match="cfg_step_lora.safetensors"):
        merge_lora(dit, variant_dir, "cfg_step_lora")


def test_merge_lora_no_modules_match_raises(tmp_path: pathlib.Path):
    """LoRA targets a path that DOESN'T exist in the DiT → AssertionError.
    Without this fail-loud check, the inference would silently produce
    wrong output."""
    from _common import merge_lora

    dit = _tiny_dit_with_one_target()
    variant_dir = tmp_path / "LongCat-Video-bf16"
    (variant_dir / "lora").mkdir(parents=True)

    # LoRA targets a path that doesn't exist in the tiny stub
    sd = _make_lora_state_dict_for("blocks.999.attn.qkv", in_dim=8, out_dim=24)
    lora_path = variant_dir / "lora" / "cfg_step_lora.safetensors"
    save_file(sd, str(lora_path))

    with pytest.raises(AssertionError, match="0 modules merged"):
        merge_lora(dit, variant_dir, "cfg_step_lora", verbose=False)


def test_merge_lora_successful_merge_changes_weights(tmp_path: pathlib.Path):
    """A well-targeted LoRA should actually mutate the matched weight tensor.
    Verify by snapshotting the weight before/after and asserting it changed.
    """
    from _common import merge_lora

    dit = _tiny_dit_with_one_target()
    variant_dir = tmp_path / "LongCat-Video-bf16"
    (variant_dir / "lora").mkdir(parents=True)

    # Snapshot the base weight before merge
    base_before = mx.array(dit.blocks[0].attn.qkv.weight)
    mx.eval(base_before)

    sd = _make_lora_state_dict_for("blocks.0.attn.qkv", in_dim=8, out_dim=24, rank=4)
    lora_path = variant_dir / "lora" / "cfg_step_lora.safetensors"
    save_file(sd, str(lora_path))

    result = merge_lora(dit, variant_dir, "cfg_step_lora", verbose=False)

    # 1 module applied, 0 unmapped
    assert len(result["applied"]) == 1
    assert result["applied"][0] == "blocks.0.attn.qkv"
    assert len(result["unmapped"]) == 0

    # Weight actually changed
    base_after = dit.blocks[0].attn.qkv.weight
    diff = float(mx.max(mx.abs(base_after - base_before)))
    assert diff > 0.0, "Merge didn't actually change the weight"


def test_load_lora_state_dict_returns_mlx_arrays(tmp_path: pathlib.Path):
    """Convenience: verify load_lora_state_dict returns mx.array values
    (not numpy arrays — the merge math is pure MLX)."""
    from _common import load_lora_state_dict
    sd = _make_lora_state_dict_for("blocks.0.attn.qkv", in_dim=8, out_dim=24)
    p = tmp_path / "tiny.safetensors"
    save_file(sd, str(p))

    loaded = load_lora_state_dict(p)
    for k, v in loaded.items():
        assert isinstance(v, mx.array), f"{k} is {type(v)}, not mx.array"
