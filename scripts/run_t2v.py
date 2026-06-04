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
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import (
    encode_prompts,
    load_components,
    merge_lora,
    postprocess_video,
    save_video_mp4,
)


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
    from longcat_video.pipeline_t2v import LongCatVideoT2VPipeline, T2VPipelineConfig

    # Shared load helper — same plumbing as I2V / Continuation / Refinement.
    vae, umt5, dit, variant_dir = load_components(weights_dir)

    # Build pipeline
    cfg = T2VPipelineConfig()
    pipeline = LongCatVideoT2VPipeline(vae=vae, text_encoder=umt5, dit=dit, config=cfg)

    # Optional: merge cfg_step_lora and flip to fast mode (collapsed CFG +
    # reduced step count). The LoRA was trained to absorb the CFG correction
    # term, so a single forward pass per step at scale=0 is equivalent to
    # the 2-pass baseline.
    if with_cfg_step_lora:
        merge_lora(dit, variant_dir, "cfg_step_lora")
        cfg.cfg_collapse = True
        cfg.num_sampling_steps = 8
        cfg.text_guidance_scale = 0.0
        print(f"  [cfg_step_lora] pipeline flipped to fast mode: "
              f"cfg_collapse=True, {cfg.num_sampling_steps} steps, "
              f"guidance_scale=0")

    return pipeline, cfg


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
                        help="Pre-merge cfg_step_lora for the fast path: "
                             "collapses CFG to a single forward per step + "
                             "reduces step count from 50 to 8")
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

    variant_dir = args.weights / "LongCat-Video-bf16"

    print("[2/5] Tokenizing + encoding prompt via umT5...")
    text_embeds, text_mask, uncond_embeds, uncond_mask = encode_prompts(
        pipeline.text_encoder, args.prompt, args.negative_prompt, variant_dir,
    )

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
    arr = postprocess_video(video)
    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), arr)
    print(f"  saved frames to {npy_path}  (shape {arr.shape})")

    print("[5/5] Encoding MP4...")
    save_video_mp4(arr, args.out, fps=pipeline.config.target_fps)
    if args.out.exists():
        print(f"  saved video to {args.out}")


if __name__ == "__main__":
    main()
