"""Weight conversion recipe for the BASE LongCat-Video model.

Mirrors the structure of `longcat-avatar-mlx/recipes/convert_longcat_avatar.py`
(same helpers, same materialize-before-save discipline) but targets the
base model's HF repo `meituan-longcat/LongCat-Video` and the different
LoRA setup.

Produces THREE publishable variants in per-component HF subdir layout:

    mlx-community/LongCat-Video-bf16/                          (~42 GB)
        vae/diffusion_pytorch_model.safetensors                (~254 MB bf16)
        text_encoder/{...sharded umT5 bf16...}                 (~11 GB)
        dit/{...sharded base DiT bf16...}                      (~26 GB)
        lora/cfg_step_lora.safetensors                         (~2.3 GB)
        lora/refinement_lora.safetensors                       (~3.0 GB)
        scheduler/scheduler_config.json
        tokenizer/                                             (umT5 SentencePiece)
        pipeline_config.json
        README.md

    mlx-community/LongCat-Video-q4/                            (~22 GB)
        dit/                                                   (~7 GB q4)
        (VAE / umT5 / LoRAs / scheduler / tokenizer same as bf16)

    mlx-community/LongCat-Video-q8/                            (~30 GB)
        dit/                                                   (~13 GB q8)
        (same)

End users call the pipelines with optional LoRA merge:
- T2V baseline: load DiT alone (50-step Flow Matching)
- T2V fast: merge `cfg_step_lora` for collapsed CFG + 8-step inference
- 720p refinement: merge `refinement_lora` + enable BSA

The q4 and q8 variants are built **from the already-converted bf16 DiT
on disk** — they don't re-download from PT. VAE / umT5 / LoRAs stay bf16
in all three variants (quantizing them would degrade output more than
save space).

CRITICAL: every saved tensor is materialized via `mx.eval` immediately
before `mx.save_safetensors`. Lazy MLX tensors serialize as ZEROS with no
error (the mlx-porting skill's silent-killer warning).

Usage:
    .venv/bin/python -m recipes.convert_longcat_video --out <PATH> --variant bf16
    .venv/bin/python -m recipes.convert_longcat_video --out <PATH> --variant q4
    .venv/bin/python -m recipes.convert_longcat_video --out <PATH> --variant q8
    .venv/bin/python -m recipes.convert_longcat_video --out <PATH> --variant all

Requires `huggingface_hub` + `safetensors` (already in `[parity]` extras).
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
PUBLISH_REPO_Q4 = "mlx-community/LongCat-Video-q4"
PUBLISH_REPO_Q8 = "mlx-community/LongCat-Video-q8"


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

# Patterns whose Linear modules must NOT be quantized. Matches Avatar's DiT
# (same 48-block architecture) and Meituan's documented skip pattern.
# - `final_layer.linear`: Meituan's published skip (preserves output quality)
# - `t_embedder.`: TimestepEmbedder MLP — small + sensitive (Linear 256→512
#   and 512→512). Drives `adaLN_modulation` input dim — see L11/L42.
# - `y_embedder.`: CaptionEmbedder MLP — small + sensitive
# - `adaLN_modulation.`: per-block AdaLN-Zero modulation. **MUST stay fp**
#   per L11 — silent accumulation bug if quantized.
DIT_QUANT_SKIP_PATTERNS: list[str] = [
    "final_layer.linear",
    "t_embedder.",
    "y_embedder.",
    "adaLN_modulation.",
]


def _should_quantize_dit_linear(path: str, module) -> bool:
    """class_predicate for `mlx.nn.quantize` on the DiT.

    Quantizes `nn.Linear` only; skips per `DIT_QUANT_SKIP_PATTERNS`.
    """
    import mlx.nn as nn

    if not isinstance(module, nn.Linear):
        return False
    for pat in DIT_QUANT_SKIP_PATTERNS:
        if pat in path:
            return False
    return True


def _write_dit_config_with_quant(
    out_dir: pathlib.Path, bits: int, group_size: int,
) -> None:
    """Copy Meituan's `dit/config.json` then inject a `quantization` block
    so the runtime loader applies `nn.quantize` before `load_weights`.
    """
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(repo_id=BASE_REPO, filename="dit/config.json")
    cfg = json.loads(pathlib.Path(src).read_text())
    cfg["quantization"] = {
        "method": "mlx.nn.quantize",
        "bits": bits,
        "group_size": group_size,
        "skip_patterns": DIT_QUANT_SKIP_PATTERNS,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def quantize_dit_from_bf16(
    bf16_variant_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    bits: int,
    group_size: int = 64,
) -> None:
    """Quantize the already-converted bf16 DiT to `bits`-bit and write it
    to `out_dir/dit/`. Re-uses the bf16 weights on disk — does NOT re-
    download from PT.

    Args:
        bf16_variant_dir: Path to existing `LongCat-Video-bf16/` directory.
        out_dir: Path to write `LongCat-Video-q{bits}/dit/...` under.
        bits: 4 or 8.
        group_size: quantization group size. Default 64 (mlx-lm convention).
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel

    assert bits in (4, 8), f"bits must be 4 or 8, got {bits}"

    src_dit_dir = bf16_variant_dir / "dit"
    print(f"  Quantizing DiT to {bits}-bit (group_size={group_size}) from "
          f"{src_dit_dir}/ → {out_dir}/dit/")

    # 1. Construct the model from the bf16 config (no quantization block —
    # we're quantizing FROM bf16, the output config gets the quant block
    # written separately via _write_dit_config_with_quant).
    dit_cfg = json.loads((src_dit_dir / "config.json").read_text())
    dit_cfg.pop("quantization", None)   # ignore if upstream already has one
    model = LongCatVideoTransformer3DModel.from_config(dit_cfg)

    # 2. Load bf16 shards
    idx = json.loads((src_dit_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    for shard_name in sorted(set(idx["weight_map"].values())):
        model.load_weights(str(src_dit_dir / shard_name), strict=False)
    mx.eval(model.parameters())

    # 3. Quantize Linears in place (skip per DIT_QUANT_SKIP_PATTERNS)
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=_should_quantize_dit_linear,
    )

    # 4. Snapshot the now-quantized parameter tree
    quantized_sd = dict(tree_flatten(model.parameters()))
    del model

    _save_sharded_safetensors(quantized_sd, out_dir / "dit",
                              base_name="diffusion_pytorch_model")
    _write_dit_config_with_quant(out_dir / "dit", bits=bits, group_size=group_size)


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


def build_q_variant(
    out_root: pathlib.Path,
    *,
    bits: int,
    group_size: int = 64,
    skip_done: bool = True,
) -> None:
    """Build the `LongCat-Video-q{bits}` variant by quantizing FROM the
    already-converted bf16 variant on disk.

    Layout: `{out_root}/LongCat-Video-q{bits}/`. Re-uses
    `{out_root}/LongCat-Video-bf16/` as the quantization source — VAE /
    umT5 / LoRAs are simply linked / re-copied bf16 (they're small enough
    that quantizing them would degrade output more than save space).

    The bf16 variant must exist at `{out_root}/LongCat-Video-bf16/`
    before running this — build it first via `build_bf16_variant`.
    """
    bf16_dir = out_root / "LongCat-Video-bf16"
    out_dir = out_root / f"LongCat-Video-q{bits}"
    if not bf16_dir.exists():
        raise FileNotFoundError(
            f"bf16 variant not found at {bf16_dir}. Run "
            f"`build_bf16_variant` first (or `--variant bf16`)."
        )
    print(f"Building q{bits} variant → {out_dir}")
    print(f"  (quantizing from existing bf16 at {bf16_dir})")

    # Re-use bf16 components verbatim. They're identical bytes; copy or
    # link. We use copy here for portability (link would require the
    # publish step to dereference; HF upload handles symlinks but local
    # smoke tests are simpler with real files).
    import shutil

    def _copy_dir(name: str, sentinel: str):
        src = bf16_dir / name
        dst = out_dir / name
        if skip_done and (dst / sentinel).exists():
            print(f"  {name} already present — skipping copy")
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  copied {name}/ from bf16 ({sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()) / 1e9:.1f} GB)")

    _copy_dir("vae", "diffusion_pytorch_model.safetensors")
    _copy_dir("text_encoder", "model.safetensors.index.json")
    _copy_dir("tokenizer", "tokenizer.json")
    _copy_dir("scheduler", "scheduler_config.json")
    _copy_dir("lora", "cfg_step_lora.safetensors")

    # Quantize the DiT
    if skip_done and _component_done(
        out_dir, "dit", "diffusion_pytorch_model.safetensors.index.json",
    ):
        print(f"  DiT (q{bits}) already converted — skipping")
    else:
        quantize_dit_from_bf16(bf16_dir, out_dir, bits=bits, group_size=group_size)

    write_pipeline_config(out_dir)
    _write_quant_readme(out_dir, bits=bits)
    print(f"DONE: {out_dir}")


def _write_quant_readme(out_dir: pathlib.Path, *, bits: int) -> None:
    """Stamp a minimal README pointing at the full model card on HF.
    Full markdown lives in docs/model-cards/q{bits}.md; this is the
    in-variant sentinel.
    """
    readme = f"""# LongCat-Video-q{bits} (MLX)

{bits}-bit quantized variant of `mlx-community/LongCat-Video-bf16`. Same
model, same six task variants — just with the DiT Linears quantized to
{bits}-bit via `mlx.nn.quantize` for smaller-RAM Macs.

See the longcat-video-mlx repo for the full model card and inference
quick start. The runtime pipeline auto-detects the `quantization` block
in `dit/config.json` and applies `nn.quantize` before loading weights.
"""
    (out_dir / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description="Convert LongCat-Video to MLX format")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Output root directory. Per-variant subdirs are created underneath.",
    )
    parser.add_argument(
        "--variant",
        choices=["bf16", "q4", "q8", "all"],
        default="bf16",
        help="Which variant(s) to build. 'all' builds bf16 then q4 then q8.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        default=True,
        help="Skip components that already exist (resumable). Default: True.",
    )
    args = parser.parse_args()

    if args.variant in ("bf16", "all"):
        build_bf16_variant(args.out / "LongCat-Video-bf16", skip_done=args.skip_done)
    if args.variant in ("q4", "all"):
        build_q_variant(args.out, bits=4, skip_done=args.skip_done)
    if args.variant in ("q8", "all"):
        build_q_variant(args.out, bits=8, skip_done=args.skip_done)


if __name__ == "__main__":
    main()
