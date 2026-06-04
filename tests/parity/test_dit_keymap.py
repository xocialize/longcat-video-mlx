"""Keymap test for the base LongCatVideoTransformer3DModel.

Validates that every PT key in the base DiT checkpoint maps cleanly to a
parameter in our MLX model. **No weights download** — uses just the
safetensors INDEX (~90 KB).

Per our isomorphic-with-PT convention (mlx-porting skill rule), NO
rename function is needed: every class name and attribute name in
`longcat_video_dit.py` matches PT exactly, modulo the standard
`_FP32` norm-as-attribute-vs-subclass detail.

This test is **cheap and always runnable** (no `[parity]` extras needed
— just `huggingface_hub` which is a runtime dep). It catches structural
drift between our MLX model and the upstream PT checkpoint before any
forward-pass parity test would fail.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import pytest

from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DIT_CONFIG_PATH = (
    REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot"
    / "longcat-video--dit-config.json"
)

HF_REPO = "meituan-longcat/LongCat-Video"
HF_DIT_INDEX = "dit/diffusion_pytorch_model.safetensors.index.json"


def test_dit_keymap_no_rename_needed():
    """Every PT key in the base DiT checkpoint exists in our MLX model's
    `parameters()` keyset. No download — just the index.
    """
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_DIT_INDEX)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    pt_keys = sorted(weight_map.keys())

    cfg = json.loads(DIT_CONFIG_PATH.read_text())
    mx_model = LongCatVideoTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    unmapped = []
    for pt_k in pt_keys:
        if pt_k not in mx_keys:
            unmapped.append(pt_k)

    assert not unmapped, (
        f"{len(unmapped)} PT keys not in MLX model. First 20:\n  "
        + "\n  ".join(unmapped[:20])
        + (f"\n  ... {len(unmapped) - 20} more" if len(unmapped) > 20 else "")
    )

    # Sanity: base DiT empirical count is ~1022 (48 blocks + top-level).
    # Lower bound catches a truncated config; upper catches duplicated keys.
    assert 900 < len(pt_keys) < 1200, (
        f"PT key count {len(pt_keys)} unexpected — possible config drift"
    )
