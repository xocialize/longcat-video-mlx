"""Weight conversion recipe for the BASE LongCat-Video model.

Mirrors the structure of `longcat-avatar-mlx/recipes/convert_longcat_avatar.py`
(same helpers, same materialize-before-save discipline) but targets the
base model's HF repo `meituan-longcat/LongCat-Video` and the different
LoRA setup.

Produces ONE publishable variant in per-component HF subdir layout:

    mlx-community/LongCat-Video-bf16/
        vae/diffusion_pytorch_model.safetensors                ( ~254 MB bf16)
        text_encoder/{...sharded umT5 bf16...}                 (~11 GB)
        dit/{...sharded base DiT bf16...}                      (~28 GB)
        lora/cfg_step_lora.safetensors                         (~2.5 GB)
        lora/refinement_lora.safetensors                       (~3.2 GB)
        scheduler/scheduler_config.json
        tokenizer/                                             (umT5 SentencePiece)
        pipeline_config.json
        README.md

End users call the pipelines with optional LoRA merge:
- T2V baseline: load DiT alone (50-step Flow Matching)
- T2V fast: merge `cfg_step_lora` for collapsed CFG + reduced step count
- 720p refinement pass: merge `refinement_lora` after the coarse pass

Quantized (q4/q8) variants land in B5.4 — same `nn.quantize` pattern as
the Avatar port, with the same DIT_QUANT_SKIP_PATTERNS.

CRITICAL: every saved tensor is materialized via `mx.eval` immediately
before `mx.save_safetensors`. Lazy MLX tensors serialize as ZEROS with no
error (the mlx-porting skill's silent-killer warning).

Usage:
    .venv/bin/python -m recipes.convert_longcat_video --out <PATH>

Requires `huggingface_hub` + `safetensors` (already in `[parity]` extras).
Uses ~90 GB of disk total (~54 GB source DiT + ~28 GB output DiT + ~22 GB
source umT5 + ~11 GB output umT5 + LoRAs + small components).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_REPO = "meituan-longcat/LongCat-Video"
PUBLISH_REPO_BF16 = "mlx-community/LongCat-Video-bf16"


# ---------------------------------------------------------------------------
# Helpers (verbatim from convert_longcat_avatar.py — keep in sync if patched)
# ---------------------------------------------------------------------------


def _materialize_and_save(
    state_dict: dict[str, mx.array],
    out_path: pathlib.Path,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Materialize every tensor (mx.eval) and write to a single safetensors file.

    The materialization step is critical: lazy MLX tensors serialize as zeros
    with no error message (the silent killer per the mlx-porting skill).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(list(state_dict.values()))
    mx.save_safetensors(str(out_path), state_dict, metadata=metadata or {})


def _save_sharded_safetensors(
    state_dict: dict[str, mx.array],
    out_dir: pathlib.Path,
    base_name: str = "diffusion_pytorch_model",
    max_shard_size_bytes: int = 5 * 1024**3,
) -> None:
    """Save a state_dict across N shards each <= max_shard_size_bytes.

    Writes an `<base_name>.safetensors.index.json` describing the weight map.
    Each shard is `<base_name>-<i>-of-<N>.safetensors`. Materializes per shard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bytes_per_dtype = {
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.float32: 4,
        mx.int8: 1,
        mx.int32: 4,
    }

    def tensor_bytes(t: mx.array) -> int:
        nbytes = bytes_per_dtype.get(t.dtype, 4)
        for d in t.shape:
            nbytes *= d
        return nbytes

    shards: list[dict[str, mx.array]] = [{}]
    cur_bytes = 0
    for k, v in state_dict.items():
        sz = tensor_bytes(v)
        if cur_bytes + sz > max_shard_size_bytes and shards[-1]:
            shards.append({})
            cur_bytes = 0
        shards[-1][k] = v
        cur_bytes += sz

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard in enumerate(shards, start=1):
        fname = f"{base_name}-{idx:05d}-of-{n_shards:05d}.safetensors"
        out_path = out_dir / fname
        mx.eval(list(shard.values()))
        mx.save_safetensors(str(out_path), shard, metadata={"format": "mlx"})
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += tensor_bytes(v)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out_dir / f"{base_name}.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _copy_meituan_config(repo_id: str, src_path: str, out_path: pathlib.Path) -> None:
    """Download + copy a config.json (or similar) verbatim from HF."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=src_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, out_path)


def _load_pt_safetensors(repo_id: str, filename: str) -> dict[str, mx.array]:
    """Single-file safetensors loader via HF + mx.load (bf16-native)."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=filename)
    return mx.load(p)


def _load_pt_safetensors_sharded(repo_id: str, index_filename: str) -> dict[str, mx.array]:
    """Sharded safetensors loader via the index. bf16-native."""
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    base_path = pathlib.Path(index_filename).parent

    shard_to_keys: dict[str, list[str]] = {}
    for k, shard_name in weight_map.items():
        shard_to_keys.setdefault(shard_name, []).append(k)

    sd: dict[str, mx.array] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id=repo_id, filename=str(base_path / shard_name))
        shard_data = mx.load(shard_path)
        for k in keys:
            sd[k] = shard_data[k]
    return sd


def _layout_and_cast(arr: mx.array, *, name: str, is_gamma: bool, dtype=mx.bfloat16) -> mx.array:
    """Apply Conv*d weight transpose + dtype cast to a tensor loaded via mx.load.

    Layout rules:
    - 1D / 2D: pass through (biases, gammas, Linear weights)
    - 3D: Conv1d (O, I, K) -> (O, K, I)
    - 4D: Conv2d (O, I, H, W) -> (O, H, W, I), UNLESS is_gamma=True
    - 5D: Conv3d (O, I, T, H, W) -> (O, T, H, W, I) for non-gamma

    Dtype rules:
    - `_key_should_stay_fp32(name)` → ALWAYS fp32 (force upcast if source is bf16)
    - otherwise cast to the target `dtype`
    """
    if is_gamma:
        pass
    elif arr.ndim == 3:
        arr = arr.transpose(0, 2, 1)
    elif arr.ndim == 4:
        arr = arr.transpose(0, 2, 3, 1)
    elif arr.ndim == 5:
        arr = arr.transpose(0, 2, 3, 4, 1)

    target_dtype = mx.float32 if _key_should_stay_fp32(name) else dtype
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)
    return arr


def _key_should_stay_fp32(name: str) -> bool:
    """Per CLAUDE.md L11: adaLN modulation linears need fp32 (silent
    accumulation bug in bf16). Mirrors the Avatar recipe predicate exactly.
    """
    # AdaLN_modulation Linear (per-block and final-layer)
    if "adaLN_modulation" in name:
        return True
    return False


# ---------------------------------------------------------------------------
# Component converters
# ---------------------------------------------------------------------------


def convert_vae(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video/vae → out_dir/vae/

    Verbatim Wan 2.1 VAE — same as Avatar's VAE.
    """
    print(f"  Converting VAE → {out_dir}/vae/")
    src = _load_pt_safetensors(BASE_REPO, "vae/diffusion_pytorch_model.safetensors")
    mlx_sd = {k: _layout_and_cast(v, name=k, is_gamma="gamma" in k, dtype=dtype) for k, v in src.items()}
    _materialize_and_save(mlx_sd, out_dir / "vae" / "diffusion_pytorch_model.safetensors")
    _copy_meituan_config(BASE_REPO, "vae/config.json", out_dir / "vae" / "config.json")


def convert_umt5(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video/text_encoder → out_dir/text_encoder/

    Loads the SHARDED PT safetensors (5 fp32 shards), applies the HF→mlx-video
    key rename used by the Avatar port, casts to bf16, re-shards.
    """
    print(f"  Converting umT5 → {out_dir}/text_encoder/")
    from longcat_video.models.umt5 import rename_pt_to_mx

    src = _load_pt_safetensors_sharded(BASE_REPO, "text_encoder/model.safetensors.index.json")
    mlx_sd: dict[str, mx.array] = {}
    for k, v in src.items():
        new_key = rename_pt_to_mx(k)
        mlx_sd[new_key] = _layout_and_cast(v, name=new_key, is_gamma="gamma" in new_key, dtype=dtype)
    _save_sharded_safetensors(mlx_sd, out_dir / "text_encoder", base_name="model")
    _copy_meituan_config(BASE_REPO, "text_encoder/config.json", out_dir / "text_encoder" / "config.json")


def convert_dit(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video/dit → out_dir/dit/

    Base 48-block DiT. fp32 source → bf16 output. No LoRA merge — both
    `cfg_step_lora` and `refinement_lora` ship as separate files at
    `out_dir/lora/` so users can load them per task variant.
    """
    print(f"  Converting base DiT → {out_dir}/dit/")
    src = _load_pt_safetensors_sharded(BASE_REPO, "dit/diffusion_pytorch_model.safetensors.index.json")
    mlx_sd: dict[str, mx.array] = {}
    for k, v in src.items():
        is_gamma = "gamma" in k
        mlx_sd[k] = _layout_and_cast(v, name=k, is_gamma=is_gamma, dtype=dtype)
    _save_sharded_safetensors(mlx_sd, out_dir / "dit", base_name="diffusion_pytorch_model")
    _copy_meituan_config(BASE_REPO, "dit/config.json", out_dir / "dit" / "config.json")


def convert_lora(out_dir: pathlib.Path, name: str) -> None:
    """meituan-longcat/LongCat-Video/lora/<name>.safetensors → out_dir/lora/<name>.safetensors

    Re-saves the LoRA with Meituan's encoded names preserved (the runtime
    loader's `lora.decode_module_name` handles decoding). Source is fp32;
    we keep it that way — LoRAs are small and the extra precision matters
    for the merge math at low rank.

    name: "cfg_step_lora" or "refinement_lora"
    """
    print(f"  Converting LoRA → {out_dir}/lora/{name}.safetensors")
    src = _load_pt_safetensors(BASE_REPO, f"lora/{name}.safetensors")
    _materialize_and_save(src, out_dir / "lora" / f"{name}.safetensors")


def copy_scheduler_and_tokenizer(out_dir: pathlib.Path) -> None:
    """Verbatim copy of scheduler config + umT5 tokenizer files."""
    print(f"  Copying scheduler + tokenizer → {out_dir}/")
    _copy_meituan_config(BASE_REPO, "scheduler/scheduler_config.json",
                         out_dir / "scheduler" / "scheduler_config.json")
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        try:
            _copy_meituan_config(BASE_REPO, f"tokenizer/{fname}", out_dir / "tokenizer" / fname)
        except Exception:
            # Not all tokenizer files exist in every repo; skip silently
            pass


def write_pipeline_config(out_dir: pathlib.Path) -> None:
    """Emit a minimal pipeline_config.json describing what's in this variant."""
    cfg = {
        "_class_name": "LongCatVideoPipeline",
        "_diffusers_version": "0.32.0",
        "variant": "bf16",
        "loras": ["cfg_step_lora", "refinement_lora"],
        "components": {
            "vae": ["AutoencoderKLWan", "diffusers"],
            "text_encoder": ["UMT5EncoderModel", "longcat_video.models.umt5"],
            "dit": ["LongCatVideoTransformer3DModel", "longcat_video.models.longcat_video_dit"],
            "scheduler": ["FlowMatchEulerDiscreteScheduler", "diffusers"],
            "tokenizer": ["T5TokenizerFast", "transformers"],
        },
    }
    (out_dir / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))


def write_readme(out_dir: pathlib.Path) -> None:
    """Stamp a minimal README.md at the variant root. Full model card is
    in docs/model-cards/ for HF publish; this one is the local sentinel.
    """
    readme = f"""# LongCat-Video-bf16 (MLX)

bf16 conversion of [meituan-longcat/LongCat-Video](https://huggingface.co/{BASE_REPO})
for inference on Apple Silicon via the
[longcat-video-mlx](https://github.com/xocialize/longcat-video-mlx) package.

Bundles VAE + umT5 + base DiT + 2 LoRAs (cfg_step + refinement) +
scheduler + tokenizer.

See the longcat-video-mlx repo for full model card, parity numbers, and
inference quick start.
"""
    (out_dir / "README.md").write_text(readme)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _component_done(out_dir: pathlib.Path, sub: str, sentinel: str) -> bool:
    """Skip-done check — same pattern as Avatar's recipe."""
    return (out_dir / sub / sentinel).exists()


def build_bf16_variant(out_dir: pathlib.Path, *, skip_done: bool = True) -> None:
    """Build the `LongCat-Video-bf16` variant."""
    print(f"Building bf16 variant → {out_dir}")
    if skip_done and _component_done(out_dir, "vae", "diffusion_pytorch_model.safetensors"):
        print("  VAE already converted — skipping")
    else:
        convert_vae(out_dir)

    if skip_done and _component_done(out_dir, "text_encoder", "model.safetensors.index.json"):
        print("  umT5 already converted — skipping")
    else:
        convert_umt5(out_dir)

    if skip_done and _component_done(out_dir, "dit", "diffusion_pytorch_model.safetensors.index.json"):
        print("  base DiT already converted — skipping")
    else:
        convert_dit(out_dir)

    if skip_done and _component_done(out_dir, "lora", "cfg_step_lora.safetensors"):
        print("  cfg_step_lora already converted — skipping")
    else:
        convert_lora(out_dir, "cfg_step_lora")

    if skip_done and _component_done(out_dir, "lora", "refinement_lora.safetensors"):
        print("  refinement_lora already converted — skipping")
    else:
        convert_lora(out_dir, "refinement_lora")

    copy_scheduler_and_tokenizer(out_dir)
    write_pipeline_config(out_dir)
    write_readme(out_dir)
    print(f"DONE: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LongCat-Video to MLX format (bf16)")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Output root directory. The bf16 variant subdir is created underneath.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        default=True,
        help="Skip components that already exist (resumable). Default: True.",
    )
    args = parser.parse_args()
    build_bf16_variant(args.out / "LongCat-Video-bf16", skip_done=args.skip_done)


if __name__ == "__main__":
    main()
