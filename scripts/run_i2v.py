"""End-to-end I2V inference: reference image + prompt → 480p video.

Loads the converted bf16 weights, loads + preprocesses the reference image,
tokenizes + encodes the prompt via umT5, runs the I2V pipeline, and saves
an MP4 (+ .npy sidecar).

Usage:
    .venv/bin/python scripts/run_i2v.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --image input.png \\
        --prompt "The cat begins to surf, sunset, cinematic" \\
        --num-frames 24 --height 480 --width 832 \\
        --out output_i2v.mp4

Modes:
- BASELINE (default): 50-step Flow Matching with text-CFG (scale=5.0).
- FAST (`--cfg-step-lora`): pre-merge `cfg_step_lora` for the fast path.
  Same WIP caveat as `run_t2v.py` — flag exists, merge wiring lands in B1.5.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import mlx.core as mx
import numpy as np

# Allow `python scripts/run_i2v.py` from the repo root
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import (
    encode_prompts,
    load_components,
    load_image,
    postprocess_video,
    save_video_mp4,
)


def build_pipeline(weights_dir: pathlib.Path, with_cfg_step_lora: bool):
    """Load components + wire the I2V pipeline."""
    from longcat_video.pipeline_i2v import I2VPipelineConfig, LongCatVideoI2VPipeline

    vae, umt5, dit, variant_dir = load_components(weights_dir)

    cfg = I2VPipelineConfig()
    pipeline = LongCatVideoI2VPipeline(
        vae=vae, text_encoder=umt5, dit=dit, config=cfg,
    )

    if with_cfg_step_lora:
        print("  cfg_step_lora flag set — merge wiring lands in B1.5; "
              "ignoring for now and using baseline 50-step CFG path.")
        # Same TODO as run_t2v.py — once B1.5 wires the merge, flip:
        # pipeline.config.cfg_collapse = True
        # pipeline.config.num_sampling_steps = 8

    return pipeline, cfg, variant_dir


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video I2V inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    parser.add_argument("--image", type=pathlib.Path, required=True,
                        help="Path to reference image (motion anchor)")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt describing the motion/scene")
    parser.add_argument("--negative-prompt", default="",
                        help="Negative/unconditional prompt (default: empty)")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-step-lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_i2v.mp4"))
    args = parser.parse_args()

    print("=== LongCat-Video I2V MLX inference ===")
    print(f"Image : {args.image}")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Output: {args.out}")
    print()

    print("[1/6] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline, cfg, variant_dir = build_pipeline(
        args.weights, with_cfg_step_lora=args.cfg_step_lora,
    )
    if args.num_steps:
        pipeline.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print("[2/6] Loading + preprocessing reference image...")
    image = load_image(args.image, args.height, args.width)

    print("[3/6] Tokenizing + encoding prompt via umT5...")
    text_embeds, text_mask, uncond_embeds, uncond_mask = encode_prompts(
        pipeline.text_encoder, args.prompt, args.negative_prompt, variant_dir,
    )

    print(f"[4/6] Running denoising loop ({pipeline.config.num_sampling_steps} steps)...")
    t1 = time.time()
    video = pipeline(
        image=image,
        text_embeds=text_embeds, text_mask=text_mask,
        uncond_embeds=uncond_embeds, uncond_mask=uncond_mask,
        num_frames=args.num_frames,
        height=args.height, width=args.width,
        seed=args.seed,
    )
    mx.eval(video)
    elapsed = time.time() - t1
    print(f"  inference: {elapsed:.1f}s ({elapsed / args.num_frames * 1000:.0f} ms/frame)")

    print("[5/6] Postprocessing...")
    arr = postprocess_video(video)
    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), arr)
    print(f"  saved frames to {npy_path}  (shape {arr.shape})")

    print("[6/6] Encoding MP4...")
    save_video_mp4(arr, args.out, fps=pipeline.config.target_fps)
    if args.out.exists():
        print(f"  saved video to {args.out}")


if __name__ == "__main__":
    main()
