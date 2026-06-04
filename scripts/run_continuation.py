"""End-to-end Video Continuation inference:
prior video clip + prompt → 480p continuation video.

Loads the converted bf16 weights, loads + preprocesses the last N frames
of an input video, tokenizes + encodes the prompt via umT5, runs the
Continuation pipeline, and saves an MP4 (+ .npy sidecar).

Usage:
    .venv/bin/python scripts/run_continuation.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --prior-video prior.mp4 \\
        --prior-num-frames 8 \\
        --prompt "Continuation: the cat dives into the wave" \\
        --num-new-frames 24 --height 480 --width 832 \\
        --out output_continuation.mp4

Modes:
- BASELINE (default): 50-step Flow Matching with text-CFG (scale=5.0).
- FAST (`--cfg-step-lora`): pre-merge `cfg_step_lora` for the fast path.
  WIP — see `run_t2v.py` and `run_i2v.py`.

The output by default includes the cond prefix (continuous video). Pass
`--strip-prefix` to drop the prefix from the decoded output (useful for
the Long-Video chaining orchestrator — B5.1 — which stitches segments).
"""

from __future__ import annotations

import argparse
import pathlib
import time

import mlx.core as mx
import numpy as np

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import (
    encode_prompts,
    load_components,
    load_video,
    merge_lora,
    postprocess_video,
    save_video_mp4,
)


def build_pipeline(weights_dir: pathlib.Path, with_cfg_step_lora: bool):
    from longcat_video.pipeline_continuation import (
        ContinuationPipelineConfig,
        LongCatVideoContinuationPipeline,
    )

    vae, umt5, dit, variant_dir = load_components(weights_dir)

    cfg = ContinuationPipelineConfig()
    pipeline = LongCatVideoContinuationPipeline(
        vae=vae, text_encoder=umt5, dit=dit, config=cfg,
    )

    if with_cfg_step_lora:
        merge_lora(dit, variant_dir, "cfg_step_lora")
        cfg.cfg_collapse = True
        cfg.num_sampling_steps = 8
        cfg.text_guidance_scale = 0.0
        print(f"  [cfg_step_lora] pipeline flipped to fast mode: "
              f"cfg_collapse=True, {cfg.num_sampling_steps} steps, "
              f"guidance_scale=0")

    return pipeline, cfg, variant_dir


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Continuation inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    parser.add_argument("--prior-video", type=pathlib.Path, required=True,
                        help="Path to prior video clip to continue from")
    parser.add_argument("--prior-num-frames", type=int, default=8,
                        help="How many trailing frames of the prior clip to "
                             "condition on (must round to ≥1 latent frame, "
                             "i.e. ≥5 raw frames; default 8)")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt describing the continuation")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-new-frames", type=int, default=24,
                        help="Number of NEW frames to generate (the output's "
                             "tail; default 24)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-step-lora", action="store_true")
    parser.add_argument("--strip-prefix", action="store_true",
                        help="Drop the cond prefix from the decoded output "
                             "(used by the Long-Video chaining orchestrator)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_continuation.mp4"))
    args = parser.parse_args()

    print("=== LongCat-Video Continuation MLX inference ===")
    print(f"Prior : {args.prior_video} (last {args.prior_num_frames} frames)")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Output: {args.out} ({'tail-only' if args.strip_prefix else 'full incl. prefix'})")
    print()

    print("[1/6] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline, cfg, variant_dir = build_pipeline(
        args.weights, with_cfg_step_lora=args.cfg_step_lora,
    )
    if args.num_steps:
        pipeline.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print(f"[2/6] Loading + preprocessing {args.prior_num_frames} prior frames...")
    prefix = load_video(
        args.prior_video, args.prior_num_frames, args.height, args.width,
    )

    print("[3/6] Tokenizing + encoding prompt via umT5...")
    text_embeds, text_mask, uncond_embeds, uncond_mask = encode_prompts(
        pipeline.text_encoder, args.prompt, args.negative_prompt, variant_dir,
    )

    print(f"[4/6] Running denoising loop ({pipeline.config.num_sampling_steps} steps)...")
    t1 = time.time()
    video = pipeline(
        prefix_video=prefix,
        text_embeds=text_embeds, text_mask=text_mask,
        uncond_embeds=uncond_embeds, uncond_mask=uncond_mask,
        num_new_frames=args.num_new_frames,
        height=args.height, width=args.width,
        seed=args.seed,
        return_full_video=not args.strip_prefix,
    )
    mx.eval(video)
    elapsed = time.time() - t1
    out_T = int(video.shape[2])
    print(f"  inference: {elapsed:.1f}s ({elapsed / out_T * 1000:.0f} ms/frame; "
          f"{out_T} frames out)")

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
