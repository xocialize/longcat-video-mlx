"""Shared helpers for the LongCat-Video MLX CLIs.

All `run_*.py` CLIs share these primitives so the per-CLI files stay thin:

1. Load the converted bf16 weights (VAE / umT5 / DiT — all three sharded).
2. Tokenize + encode prompts via the umT5 tokenizer.
3. Encode an image (I2V) or video clip (Continuation) into VAE latents.
4. **Hot-swap-merge a LoRA** into the DiT (`cfg_step_lora`,
   `refinement_lora`) before inference — one helper, one call.
5. Save the decoded video as an .mp4 (+ .npy sidecar).

Single source of truth — if a behavior is in multiple CLIs, it lives here.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import mlx.core as mx
import numpy as np


# -------------------- Model loading --------------------------------------

def load_components(weights_dir: pathlib.Path):
    """Load all three components (VAE, umT5, DiT) from the converted bf16
    directory layout produced by `recipes/convert_longcat_video.py`.

    Returns `(vae, umt5, dit, variant_dir)`. The caller wires them into
    whichever pipeline (T2V / I2V / Continuation / Refinement).
    """
    from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.models.umt5 import UMT5EncoderModel

    variant_dir = weights_dir / "LongCat-Video-bf16"
    print(f"Loading from {variant_dir}")

    # VAE — single file
    vae_cfg = json.loads((variant_dir / "vae" / "config.json").read_text())
    vae = AutoencoderKLWan.from_config(vae_cfg)
    vae.load_weights(
        str(variant_dir / "vae" / "diffusion_pytorch_model.safetensors"),
        strict=False,
    )

    # umT5 (sharded)
    umt5_cfg = json.loads((variant_dir / "text_encoder" / "config.json").read_text())
    umt5 = UMT5EncoderModel.from_config(umt5_cfg)
    umt5_idx = json.loads(
        (variant_dir / "text_encoder" / "model.safetensors.index.json").read_text()
    )
    for shard_name in sorted(set(umt5_idx["weight_map"].values())):
        umt5.load_weights(
            str(variant_dir / "text_encoder" / shard_name), strict=False,
        )

    # DiT (sharded)
    dit_cfg = json.loads((variant_dir / "dit" / "config.json").read_text())
    dit = LongCatVideoTransformer3DModel.from_config(dit_cfg)
    dit_idx = json.loads(
        (variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    for shard_name in sorted(set(dit_idx["weight_map"].values())):
        dit.load_weights(
            str(variant_dir / "dit" / shard_name), strict=False,
        )

    mx.eval(vae.parameters(), umt5.parameters(), dit.parameters())
    return vae, umt5, dit, variant_dir


# -------------------- Tokenization --------------------------------------

def tokenize_prompt(prompt: str, variant_dir: pathlib.Path) -> tuple[mx.array, mx.array]:
    """Tokenize via the umT5 (T5 SentencePiece) tokenizer.

    Returns `(input_ids [1, 512], attention_mask [1, 512])`.
    """
    try:
        from transformers import T5TokenizerFast
    except ImportError as e:
        raise ImportError(
            "transformers is required for tokenization. "
            "`pip install -e \".[parity]\"` (or `pip install transformers`)"
        ) from e

    tok = T5TokenizerFast.from_pretrained(str(variant_dir / "tokenizer"))
    enc = tok(prompt, return_tensors="np", padding="max_length",
              max_length=512, truncation=True)
    return mx.array(enc.input_ids), mx.array(enc.attention_mask)


def encode_prompts(
    text_encoder, prompt: str, negative_prompt: str, variant_dir: pathlib.Path
):
    """Tokenize + encode both the positive and negative prompts.

    Returns `(text_embeds, text_mask, uncond_embeds, uncond_mask)` with
    the extra singleton dim that the DiT expects (`[B, 1, N_text, 4096]`).
    """
    ids, mask = tokenize_prompt(prompt, variant_dir)
    text_hidden = text_encoder(ids, mask=mask)
    text_embeds = text_hidden[:, None, :, :]
    text_mask = mask[:, None, None, :]

    if negative_prompt:
        ids_neg, mask_neg = tokenize_prompt(negative_prompt, variant_dir)
        uncond_hidden = text_encoder(ids_neg, mask=mask_neg)
        uncond_embeds = uncond_hidden[:, None, :, :]
        uncond_mask = mask_neg[:, None, None, :]
    else:
        empty_ids = mx.zeros_like(ids)
        empty_mask = mx.zeros_like(mask)
        uncond_hidden = text_encoder(empty_ids, mask=empty_mask)
        uncond_embeds = uncond_hidden[:, None, :, :]
        uncond_mask = empty_mask[:, None, None, :]

    return text_embeds, text_mask, uncond_embeds, uncond_mask


# -------------------- Image / video I/O ---------------------------------

def load_image(path: pathlib.Path, height: int, width: int) -> mx.array:
    """Load an image, center-crop+resize to (height, width), normalize to
    [-1, 1], return shape `[1, 3, 1, H, W]` (single-frame video tensor).
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow required for I2V image loading") from e

    img = Image.open(str(path)).convert("RGB")
    # Letterbox-style fit: resize so the short side covers, center-crop
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    arr = arr.transpose(2, 0, 1)[None, :, None, :, :]      # [1, 3, 1, H, W]
    return mx.array(arr)


def load_video(
    path: pathlib.Path, num_frames: int, height: int, width: int
) -> mx.array:
    """Load up to `num_frames` frames from `path` (last-N if longer),
    center-crop+resize each to (H, W), normalize to [-1, 1], return
    `[1, 3, T, H, W]`.

    For the Continuation pipeline: pass `num_frames` matching what the
    pipeline expects (typically a small prefix like 8 frames).
    """
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise ImportError("imageio[ffmpeg] required for video loading") from e

    frames = list(iio.imiter(str(path), plugin="pyav"))
    if len(frames) > num_frames:
        frames = frames[-num_frames:]
    elif len(frames) < num_frames:
        raise ValueError(
            f"Video {path} has only {len(frames)} frames; need {num_frames}"
        )

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow required for video resize") from e

    processed = []
    for f in frames:
        img = Image.fromarray(f).convert("RGB")
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        img = img.crop((left, top, left + width, top + height))
        processed.append(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)

    arr = np.stack(processed, axis=0)            # [T, H, W, 3]
    arr = arr.transpose(3, 0, 1, 2)[None, :]     # [1, 3, T, H, W]
    return mx.array(arr)


def save_video_mp4(
    video_arr: np.ndarray, out_path: pathlib.Path, fps: int = 15
) -> None:
    """video_arr: [T, H, W, 3] uint8. Writes an MP4 via imageio."""
    try:
        import imageio
    except ImportError:
        print("  (imageio missing — skipping MP4. Use .npy fallback.)")
        return
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8
    )
    for frame in video_arr:
        writer.append_data(frame)
    writer.close()


def postprocess_video(video: mx.array) -> np.ndarray:
    """[1, 3, T, H, W] in [-1, 1] → [T, H, W, 3] uint8.

    `transpose(0, 2, 3, 4, 1)` after un-batching → `[1, T, H, W, 3]` → `[0]`.
    """
    return (
        np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5
    ).clip(0, 255).astype(np.uint8)


# -------------------- LoRA merge ----------------------------------------

def load_lora_state_dict(lora_path: pathlib.Path) -> dict:
    """Load a `.safetensors` LoRA file into an MLX state dict.

    Returns `{key: mx.array}` ready for `merge_lora_into_model`.

    Uses the numpy framework backend so we don't pull in torch at runtime.
    """
    from safetensors import safe_open

    if not lora_path.exists():
        raise FileNotFoundError(
            f"LoRA file not found: {lora_path}\n"
            f"Run the conversion recipe first: "
            f"`python -m recipes.convert_longcat_video --out <PATH>`"
        )

    state_dict: dict[str, mx.array] = {}
    with safe_open(str(lora_path), framework="numpy") as f:
        for k in f.keys():
            state_dict[k] = mx.array(f.get_tensor(k))
    return state_dict


def merge_lora(
    dit,
    variant_dir: pathlib.Path,
    name: str,
    multiplier: float = 1.0,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Convenience wrapper: load + merge a LoRA into the DiT in one call.

    Args:
        dit: the MLX DiT module (`LongCatVideoTransformer3DModel` instance).
        variant_dir: path to `LongCat-Video-bf16/` (parent of `lora/`).
        name: `"cfg_step_lora"` or `"refinement_lora"`.
        multiplier: per-LoRA strength (default 1.0; matches PT runtime).
        verbose: print applied/unmapped counts.

    Returns: `{"applied": [...], "unmapped": [...]}` from
        `merge_lora_into_model` — useful for the caller to assert
        no surprises.

    Raises:
        FileNotFoundError: if the LoRA file is missing.
        AssertionError: if zero modules were merged (something is wrong
            with the LoRA file or the DiT module tree — a no-op merge
            would silently produce wrong outputs, so we fail loudly).
    """
    from longcat_video.lora import merge_lora_into_model

    lora_path = variant_dir / "lora" / f"{name}.safetensors"
    if verbose:
        print(f"  [{name}] loading {lora_path.name} ({lora_path.stat().st_size / 1e9:.1f} GB)...")
    sd = load_lora_state_dict(lora_path)

    if verbose:
        print(f"  [{name}] merging {len(sd)} tensors into DiT...")
    result = merge_lora_into_model(dit, sd, multiplier=multiplier)

    n_applied = len(result["applied"])
    n_unmapped = len(result["unmapped"])
    if verbose:
        print(f"  [{name}] merged {n_applied} modules, {n_unmapped} unmapped")
        if n_unmapped and n_unmapped < 10:
            for path in result["unmapped"]:
                print(f"      unmapped: {path}")
        elif n_unmapped:
            for path in result["unmapped"][:5]:
                print(f"      unmapped: {path}")
            print(f"      ... and {n_unmapped - 5} more")

    assert n_applied > 0, (
        f"{name}: 0 modules merged — LoRA target paths don't match the "
        f"DiT module tree. A no-op merge would silently produce wrong "
        f"output, so failing loudly. Check lora.decode_module_name's "
        f"output against dict(tree_flatten(dit.parameters())).keys()."
    )

    # Free the LoRA state dict before returning — the delta is already in
    # the DiT weights now, so the LoRA arrays are no longer needed and
    # can release ~2-3 GB of unified memory.
    del sd
    return result
