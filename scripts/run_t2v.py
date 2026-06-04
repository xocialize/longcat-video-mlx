"""End-to-end T2V inference: prompt → 480p video.

Loads the converted bf16 weights, tokenizes the prompt via the umT5
tokenizer, encodes it, runs the T2V pipeline, saves an MP4 (and a .npy
sidecar for downstream tooling / parity tests).

Companion to longcat-avatar-mlx/scripts/run_inference.py.

Usage:
    .venv/bin/python scripts/run_t2v.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --prompt "A cat surfing on a wave at sunset, cinematic, 8k" \\
        --num-frames 24 --height 480 --width 832 \\
        --out output.mp4

Modes:
- BASELINE (default): 50-step Flow Matching with text-CFG (scale=5.0).
  Two DiT forward passes per step (cond + uncond).
- FAST (`--cfg-step-lora`): pre-merge `cfg_step_lora` into the DiT,
  collapses CFG into a single forward per step. Use --num-steps 8 or
  similar (recipe-determined).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import mlx.core as mx
import numpy as np


def build_pipeline(weights_dir: pathlib.Path, with_cfg_step_lora: bool):
    """Load all components from the converted bf16 dir + wire the pipeline.

    Expected layout (matches recipes/convert_longcat_video.py output):
        weights_dir/LongCat-Video-bf16/
            vae/
            text_encoder/
            dit/
            lora/cfg_step_lora.safetensors
            lora/refinement_lora.safetensors
            ...
    """
    from longcat_video.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.models.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.models.umt5 import UMT5EncoderModel
    from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline, T2VPipelineConfig

    variant_dir = weights_dir / "LongCat-Video-bf16"
    print(f"Loading from {variant_dir}")

    # VAE
    vae_cfg = json.loads((variant_dir / "vae" / "config.json").read_text())
    vae = AutoencoderKLWan.from_config(vae_cfg)
    vae.load_weights(str(variant_dir / "vae" / "diffusion_pytorch_model.safetensors"),
                     strict=False)

    # umT5 (sharded)
    umt5_cfg = json.loads((variant_dir / "text_encoder" / "config.json").read_text())
    umt5 = UMT5EncoderModel.from_config(umt5_cfg)
    umt5_idx = json.loads(
        (variant_dir / "text_encoder" / "model.safetensors.index.json").read_text()
    )
    for shard_name in sorted(set(umt5_idx["weight_map"].values())):
        umt5.load_weights(str(variant_dir / "text_encoder" / shard_name), strict=False)

    # DiT (sharded)
    dit_cfg = json.loads((variant_dir / "dit" / "config.json").read_text())
    dit = LongCatVideoTransformer3DModel.from_config(dit_cfg)
    dit_idx = json.loads(
        (variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    for shard_name in sorted(set(dit_idx["weight_map"].values())):
        dit.load_weights(str(variant_dir / "dit" / shard_name), strict=False)

    mx.eval(vae.parameters(), umt5.parameters(), dit.parameters())

    # Build pipeline
    cfg = T2VPipelineConfig()
    pipeline = LongCatVideoT2VPipeline(vae=vae, text_encoder=umt5, dit=dit, config=cfg)

    # Optional: merge cfg_step_lora for the fast path
    if with_cfg_step_lora:
        print("  Merging cfg_step_lora...")
        from safetensors import safe_open

        from longcat_video.lora import compute_merged_delta, group_lora_tensors

        lora_path = variant_dir / "lora" / "cfg_step_lora.safetensors"
        lora_sd = {}
        with safe_open(str(lora_path), framework="numpy") as f:
            for k in f.keys():
                lora_sd[k] = mx.array(f.get_tensor(k))

        grouped = group_lora_tensors(lora_sd)
        merged_count = 0
        for module_path, group in grouped.items():
            # Walk the dit parameters to find the matching weight tensor
            # For now use the same approach as Avatar's merge_dmd_lora —
            # need to access the dit's state_dict-equivalent.
            # Defer the actual merge to a follow-up (B1.5 will harden this).
            merged_count += 1
        print(f"  Identified {merged_count} cfg_step_lora target modules. "
              "Full merge wiring lands in B1.5 — for now the LoRA is loaded "
              "but not applied. Using baseline 50-step path.")
        # Flip into fast mode regardless so the CLI flag has visible effect
        # once B1.5 wires the actual merge:
        # pipeline.config.cfg_collapse = True
        # pipeline.config.num_sampling_steps = 8

    return pipeline, cfg


def tokenize_prompt(prompt: str, weights_dir: pathlib.Path) -> tuple[mx.array, mx.array]:
    """Tokenize via the umT5 tokenizer (T5 sentencepiece). Returns
    (input_ids [1, 512], attention_mask [1, 512])."""
    try:
        from transformers import T5TokenizerFast
    except ImportError as e:
        raise ImportError(
            "transformers is required for tokenization. "
            "`pip install -e \".[parity]\"` (or `pip install transformers`)"
        ) from e

    tok_dir = weights_dir / "LongCat-Video-bf16" / "tokenizer"
    tok = T5TokenizerFast.from_pretrained(str(tok_dir))
    enc = tok(prompt, return_tensors="np", padding="max_length",
              max_length=512, truncation=True)
    ids = mx.array(enc.input_ids)
    mask = mx.array(enc.attention_mask)
    return ids, mask


def save_video_mp4(video_arr: np.ndarray, out_path: pathlib.Path, fps: int = 15) -> None:
    """video_arr: [T, H, W, 3] uint8. Writes an MP4 via imageio."""
    try:
        import imageio
    except ImportError:
        print(f"  (imageio missing — skipping MP4. Use .npy fallback.)")
        return
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for frame in video_arr:
        writer.append_data(frame)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video T2V inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt for the video")
    parser.add_argument("--negative-prompt", default="",
                        help="Negative/unconditional prompt (default: empty)")
    parser.add_argument("--num-frames", type=int, default=24,
                        help="Frames (24 @ 15fps = ~1.6s coarse output)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override pipeline default (50 baseline / ~8 with cfg_step_lora)")
    parser.add_argument("--cfg-step-lora", action="store_true",
                        help="Pre-merge cfg_step_lora for the fast path "
                             "(WIP — see B1.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_t2v.mp4"))
    args = parser.parse_args()

    print("=== LongCat-Video T2V MLX inference ===")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Output: {args.out}")
    print()

    print("[1/5] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline, cfg = build_pipeline(args.weights, with_cfg_step_lora=args.cfg_step_lora)
    if args.num_steps:
        pipeline.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print("[2/5] Tokenizing + encoding prompt via umT5...")
    ids, mask = tokenize_prompt(args.prompt, args.weights)
    text_hidden = pipeline.text_encoder(ids, mask=mask)
    text_embeds = text_hidden[:, None, :, :]    # [B, 1, N_text, C]
    text_mask = mask[:, None, None, :]

    if args.negative_prompt:
        ids_neg, mask_neg = tokenize_prompt(args.negative_prompt, args.weights)
        uncond_hidden = pipeline.text_encoder(ids_neg, mask=mask_neg)
        uncond_embeds = uncond_hidden[:, None, :, :]
        uncond_mask = mask_neg[:, None, None, :]
    else:
        empty_ids = mx.zeros_like(ids)
        empty_mask = mx.zeros_like(mask)
        uncond_hidden = pipeline.text_encoder(empty_ids, mask=empty_mask)
        uncond_embeds = uncond_hidden[:, None, :, :]
        uncond_mask = empty_mask[:, None, None, :]

    print(f"[3/5] Running denoising loop ({pipeline.config.num_sampling_steps} steps)...")
    t1 = time.time()
    video = pipeline(
        text_embeds=text_embeds,
        text_mask=text_mask,
        uncond_embeds=uncond_embeds,
        uncond_mask=uncond_mask,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    mx.eval(video)
    elapsed = time.time() - t1
    print(f"  inference: {elapsed:.1f}s ({elapsed / args.num_frames * 1000:.0f} ms/frame)")

    print("[4/5] Postprocessing...")
    arr = (np.asarray(video).transpose(0, 2, 3, 4, 1)[0]
           * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), arr)
    print(f"  saved frames to {npy_path}  (shape {arr.shape})")

    print("[5/5] Encoding MP4...")
    save_video_mp4(arr, args.out, fps=pipeline.config.target_fps)
    if args.out.exists():
        print(f"  saved video to {args.out}")


if __name__ == "__main__":
    main()
