"""PT↔MLX parity for AutoencoderKLWan.

Pass criteria (fp32 inference on CPU/Metal):
- encode: max_abs < 1e-3 vs PT reference on a seeded video input
- decode: max_abs < 2e-2 — relaxed because the Wan VAE has heavy
  large-spatial convolutions in the decoder, which compound the
  Metal-GPU fp32 precision loss documented in L11. Avatar port observed
  max_abs ≈ 1.2e-2, mean_abs ≈ 3.5e-4 (rel_err ≈ 0.7%) — well below
  perceptual threshold; the implementation is verified correct via
  CPU-stream isolated-op tests.

Skip if torch / diffusers / weights are not available — torch is an
optional `[parity]` dev dep, not a runtime dep.

Weights:
- Set `LONGCAT_VAE_WEIGHTS_DIR` to a local directory containing
  `diffusion_pytorch_model.safetensors` from
  `meituan-longcat/LongCat-Video/vae/`, OR
- Set `LONGCAT_VAE_AUTO_DOWNLOAD=1` to auto-download (~254 MB).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

try:
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        AutoencoderKLWan as PTAutoencoderKLWan,
    )
    _PT_AVAILABLE = True
except Exception as _exc:  # ImportError or version mismatch
    _PT_AVAILABLE = False
    _IMPORT_ERR = _exc

import mlx.core as mx

from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
from tests.parity._helpers import assert_parity, make_seeded_input, mx_to_np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VAE_CONFIG_PATH = (
    REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot"
    / "longcat-video--vae-config.json"
)

HF_REPO = "meituan-longcat/LongCat-Video"
HF_VAE_PATH = "vae/diffusion_pytorch_model.safetensors"


def _locate_weights() -> str | None:
    """Find the PT VAE weights file. Returns a path or None to skip."""
    env_dir = os.environ.get("LONGCAT_VAE_WEIGHTS_DIR")
    if env_dir:
        candidate = pathlib.Path(env_dir) / "diffusion_pytorch_model.safetensors"
        if candidate.exists():
            return str(candidate)

    if os.environ.get("LONGCAT_VAE_AUTO_DOWNLOAD") == "1":
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, filename=HF_VAE_PATH)

    return None


def _load_pt_vae(weights_path: str):
    cfg = json.loads(VAE_CONFIG_PATH.read_text())
    pt_vae = PTAutoencoderKLWan(
        z_dim=cfg["z_dim"],
        base_dim=cfg["base_dim"],
        dim_mult=cfg["dim_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        temperal_downsample=cfg["temperal_downsample"],
        attn_scales=cfg["attn_scales"],
        latents_mean=cfg["latents_mean"],
        latents_std=cfg["latents_std"],
    )
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    missing, unexpected = pt_vae.load_state_dict(state_dict, strict=False)
    if unexpected:
        pytest.fail(f"Unexpected PT keys: {unexpected[:5]}"
                    f"{'...' if len(unexpected) > 5 else ''}")
    pt_vae.eval()
    return pt_vae


def _load_mx_vae(weights_path: str):
    """Load Meituan's PT VAE checkpoint into our MLX AutoencoderKLWan.

    Handles PT Conv*d (O,I,*K) → MLX (O,*K,I) layout transpose.
    """
    cfg = json.loads(VAE_CONFIG_PATH.read_text())
    mx_vae = AutoencoderKLWan.from_config(cfg, encoder=True)

    from safetensors.torch import load_file
    from mlx.utils import tree_unflatten

    state_dict = load_file(weights_path)
    flat = []
    for k, v in state_dict.items():
        arr = v.detach().cpu().float().numpy()
        if "gamma" in k:
            pass  # RMS_norm gamma — keep as-is (singleton dims)
        elif arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        elif arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        flat.append((k, mx.array(arr)))

    mx_vae.update(tree_unflatten(flat))
    mx.eval(mx_vae.parameters())
    return mx_vae


@pytest.mark.skipif(
    not _PT_AVAILABLE,
    reason="parity dep missing — install with `pip install -e '.[parity]'`",
)
def test_vae_decode_parity():
    weights = _locate_weights()
    if not weights:
        pytest.skip(
            "VAE weights not located. Set LONGCAT_VAE_WEIGHTS_DIR or "
            "LONGCAT_VAE_AUTO_DOWNLOAD=1."
        )

    pt_vae = _load_pt_vae(weights)
    mx_vae = _load_mx_vae(weights)

    # Both decoders use the same raw-z convention (no input normalization).
    z_np = make_seeded_input((1, 16, 3, 16, 16), seed=42).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_vae.decode(torch.from_numpy(z_np)).sample
    mx_out = mx_vae.decode(mx.array(z_np))

    assert_parity(pt_out, mx_out, threshold=2e-2, name="vae.decode")


@pytest.mark.skipif(
    not _PT_AVAILABLE,
    reason="parity dep missing — install with `pip install -e '.[parity]'`",
)
def test_vae_encode_parity():
    weights = _locate_weights()
    if not weights:
        pytest.skip(
            "VAE weights not located. Set LONGCAT_VAE_WEIGHTS_DIR or "
            "LONGCAT_VAE_AUTO_DOWNLOAD=1."
        )

    pt_vae = _load_pt_vae(weights)
    mx_vae = _load_mx_vae(weights)

    # Seeded random video in [-1, 1], 9 frames so encode produces 3 latent frames
    v_np = make_seeded_input((1, 3, 9, 32, 32), seed=42).astype(np.float32)
    v_np = np.clip(v_np, -1.0, 1.0)

    with torch.no_grad():
        posterior = pt_vae.encode(torch.from_numpy(v_np)).latent_dist
        pt_z = posterior.mean

    mx_z = mx_vae.encode(mx.array(v_np))  # raw mean, no normalization
    assert_parity(pt_z, mx_z, threshold=1e-3, name="vae.encode")
