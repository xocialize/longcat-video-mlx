"""End-to-end Long-Video inference: prompt → ~1 minute 480p video.

Chains a T2V seed clip with N-1 Continuation segments, each conditioned
on the last K frames of the previous segment. Writes intermediate MP4s
per segment (matching upstream's demo behavior) so a long run is
resumable / inspectable.

Default config (matches upstream `run_demo_long_video.py`):
  num_segments=11, num_frames_per_segment=93, num_cond_frames=13
  → total = 93 + 10*(93-13) = 893 frames ≈ 59.5s @ 15fps

Usage:
    .venv/bin/python scripts/run_long_video.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --prompt "A cat surfing on a wave at sunset, cinematic, 8k" \\
        --num-segments 11 \\
        --out output_long_video.mp4

For a coarse-to-fine refinement pass at 720p/30fps on the resulting
long-video, chain through `scripts/run_refine.py` with the .npy sidecar
this script writes.
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
    save_video_mp4,
)


def build_pipeline(
    weights_dir: pathlib.Path,
    num_segments: int,
    num_frames_per_segment: int,
    num_cond_frames: int,
    height: int,
    width: int,
    with_cfg_step_lora: bool,
):
    from longcat_video.pipeline_long_video import (
        LongCatVideoLongVideoPipeline,
        LongVideoPipelineConfig,
    )

    vae, umt5, dit, variant_dir = load_components(weights_dir)

    cfg = LongVideoPipelineConfig(
        num_segments=num_segments,
        num_frames_per_segment=num_frames_per_segment,
        num_cond_frames=num_cond_frames,
        height=height,
        width=width,
    )
    pipeline = LongCatVideoLongVideoPipeline(
        vae=vae, text_encoder=umt5, dit=dit, config=cfg,
    )

    if with_cfg_step_lora:
        print("  cfg_step_lora flag set — merge wiring lands in B1.5 follow-up; "
              "ignoring for now and using baseline 50-step CFG path.")

    return pipeline, cfg, variant_dir


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Long-Video inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt — same for all segments (Long-Video). "
                             "For per-segment prompts use run_interactive.py.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-segments", type=int, default=11,
                        help="11 segments × (93-13) new frames ≈ 1 min @ 15fps")
    parser.add_argument("--num-frames-per-segment", type=int, default=93)
    parser.add_argument("--num-cond-frames", type=int, default=13,
                        help="Last-N frames of each segment carried as cond "
                             "into the next (default 13)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-step-lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_long_video.mp4"))
    parser.add_argument("--write-segments", action="store_true",
                        help="Write intermediate MP4s per segment (debug)")
    args = parser.parse_args()

    expected_total = (args.num_frames_per_segment
                      + (args.num_segments - 1)
                        * (args.num_frames_per_segment - args.num_cond_frames))
    duration_s = expected_total / 15.0

    print("=== LongCat-Video Long-Video MLX inference ===")
    print(f"Prompt    : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Segments  : {args.num_segments} × {args.num_frames_per_segment} "
          f"frames (cond {args.num_cond_frames})")
    print(f"Total     : {expected_total} frames ≈ {duration_s:.1f}s @ 15fps")
    print(f"Output    : {args.out}")
    print()

    print("[1/5] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline, cfg, variant_dir = build_pipeline(
        args.weights, args.num_segments, args.num_frames_per_segment,
        args.num_cond_frames, args.height, args.width,
        with_cfg_step_lora=args.cfg_step_lora,
    )
    if args.num_steps:
        pipeline.t2v.config.num_sampling_steps = args.num_steps
        pipeline.continuation.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print("[2/5] Tokenizing + encoding prompt via umT5...")
    text_embeds, text_mask, uncond_embeds, uncond_mask = encode_prompts(
        pipeline.text_encoder, args.prompt, args.negative_prompt, variant_dir,
    )

    print(f"[3/5] Generating {args.num_segments} segments...")

    def on_done(seg_idx, frames):
        elapsed = time.time() - t1
        print(f"  segment {seg_idx + 1}/{args.num_segments} "
              f"({frames.shape[0]} new frames) — total elapsed {elapsed:.1f}s")
        if args.write_segments:
            seg_path = args.out.with_name(f"{args.out.stem}_seg{seg_idx+1:02d}.mp4")
            save_video_mp4(frames, seg_path, fps=cfg.target_fps)
            print(f"    wrote {seg_path}")

    t1 = time.time()
    all_frames = pipeline(
        text_embeds=text_embeds, text_mask=text_mask,
        uncond_embeds=uncond_embeds, uncond_mask=uncond_mask,
        num_segments=args.num_segments,
        seed=args.seed,
        on_segment_done=on_done,
    )
    elapsed = time.time() - t1
    print(f"  total inference: {elapsed:.1f}s "
          f"({elapsed / all_frames.shape[0] * 1000:.0f} ms/frame; "
          f"{all_frames.shape[0]} frames out)")

    print("[4/5] Saving .npy sidecar...")
    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), all_frames)
    print(f"  saved frames to {npy_path}  (shape {all_frames.shape})")

    print("[5/5] Encoding MP4...")
    save_video_mp4(all_frames, args.out, fps=cfg.target_fps)
    if args.out.exists():
        print(f"  saved video to {args.out}")


if __name__ == "__main__":
    main()
