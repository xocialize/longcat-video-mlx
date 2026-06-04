"""Smoke tests for the BSA enable/disable wiring on the DiT.

Verifies:
1. `enable_bsa: true` in the config flips `attn.enable_bsa = True` on
   every block.
2. `dit.enable_bsa()` and `dit.disable_bsa()` round-trip correctly.
3. BSA params (sparsity, chunk_thw) preserved from config on the DiT
   instance survive into the attention modules.
4. Default published config (which has `enable_bsa: false`) leaves BSA
   off after `from_config`.
"""

from __future__ import annotations

import json
import pathlib

import pytest


def _minimal_config(enable_bsa: bool = False) -> dict:
    """Tiny config so the DiT constructs fast in CI."""
    return {
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 64,           # tiny
        "depth": 2,                   # only 2 blocks
        "num_heads": 4,
        "caption_channels": 64,
        "mlp_ratio": 2,
        "adaln_tembed_dim": 32,
        "frequency_embedding_size": 32,
        "patch_size": [1, 2, 2],
        "text_tokens_zero_pad": False,
        "bsa_params": {
            "sparsity": 0.9375,
            "chunk_3d_shape_q": [4, 4, 4],
            "chunk_3d_shape_k": [4, 4, 4],
        },
        "enable_bsa": enable_bsa,
    }


def test_default_bsa_off():
    from longcat_video.models.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )
    cfg = _minimal_config(enable_bsa=False)
    dit = LongCatVideoTransformer3DModel.from_config(cfg)
    # Two blocks, both should have BSA off
    for blk in dit.blocks:
        assert blk.attn.enable_bsa is False


def test_config_enable_bsa_flips_attention():
    from longcat_video.models.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )
    cfg = _minimal_config(enable_bsa=True)
    dit = LongCatVideoTransformer3DModel.from_config(cfg)
    for blk in dit.blocks:
        assert blk.attn.enable_bsa is True
        assert blk.attn.bsa_sparsity == 0.9375
        assert blk.attn.bsa_chunk_thw == (4, 4, 4)


def test_dit_enable_disable_roundtrip():
    from longcat_video.models.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )
    cfg = _minimal_config(enable_bsa=False)
    dit = LongCatVideoTransformer3DModel.from_config(cfg)

    dit.enable_bsa()
    for blk in dit.blocks:
        assert blk.attn.enable_bsa is True

    dit.disable_bsa()
    for blk in dit.blocks:
        assert blk.attn.enable_bsa is False


def test_bsa_params_preserved_through_enable():
    """If bsa_params overrides defaults, the override flows through."""
    from longcat_video.models.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )
    cfg = _minimal_config(enable_bsa=False)
    cfg["bsa_params"]["sparsity"] = 0.5
    cfg["bsa_params"]["chunk_3d_shape_q"] = [2, 2, 2]
    dit = LongCatVideoTransformer3DModel.from_config(cfg)
    dit.enable_bsa()
    for blk in dit.blocks:
        assert blk.attn.bsa_sparsity == 0.5
        assert blk.attn.bsa_chunk_thw == (2, 2, 2)


def test_published_config_snapshot_has_bsa_off():
    """The published `dit/config.json` snapshot has `enable_bsa: false` —
    base T2V/I2V/Continuation must NOT enable BSA without the refinement
    pass flipping it on explicitly. Confirms the snapshot's invariant.
    """
    snap_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "docs" / "development" / "notes" / "config-snapshot"
        / "longcat-video--dit-config.json"
    )
    if not snap_path.exists():
        pytest.skip(f"config snapshot missing: {snap_path}")
    cfg = json.loads(snap_path.read_text())
    assert cfg.get("enable_bsa", True) is False, \
        "Published config must have enable_bsa: false (refinement only)"
    bsa = cfg.get("bsa_params", {})
    assert bsa.get("sparsity") == 0.9375
    assert bsa.get("chunk_3d_shape_q") == [4, 4, 4]
