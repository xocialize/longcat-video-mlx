"""End-to-end Interactive Video inference: list of prompts → chained video.

One T2V seed segment + chained Continuation segments, each conditioned
on the last K frames of the previous AND using its own prompt. Use this
when the scene evolves across the video (a dialogue, a narrative arc, a
camera trajectory). For one-prompt-per-video use `run_long_video.py`.

Prompt input:
- `--prompts-file PATH` — one prompt per line (preferred for >3 prompts)
- `--prompt P1 --prompt P2 --prompt P3 ...` — repeated arg

Usage:
    .venv/bin/python scripts/run_interactive.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --prompts-file my_story.txt \\
        --out output_interactive.mp4

    .venv/bin/python scripts/run_interactive.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --prompt "A cat enters the frame, curious" \\
        --prompt "The cat sees a butterfly and chases it" \\
        --prompt "The butterfly leads to a sunny meadow" \\
        --out output_interactive.mp4

Negative prompts:
- `--negative-prompt P` applies the SAME negative to all segments
- `--negative-prompts-file PATH` lets you provide one per segment
  (must have the same line count as the positive prompts)
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
    merge_lora,
    save_video_mp4,
)


def load_prompts(file_path: pathlib.Path) -> list[str]:
    """Read one prompt per line, skipping blanks and #-comments."""
    lines = []
    for raw in file_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_pipeline(
    weights_dir: pathlib.Path,
    num_frames_per_segment: int,
    num_cond_frames: int,
    height: int,
    width: int,
    with_cfg_step_lora: bool = False,
):
    from longcat_video.pipeline_interactive import (
        InteractivePipelineConfig,
        LongCatVideoInteractivePipeline,
    )

    vae, umt5, dit, variant_dir = load_components(weights_dir)

    cfg = InteractivePipelineConfig(
        num_frames_per_segment=num_frames_per_segment,
        num_cond_frames=num_cond_frames,
        height=height,
        width=width,
    )
    pipeline = LongCatVideoInteractivePipeline(
        vae=vae, text_encoder=umt5, dit=dit, config=cfg,
    )

    if with_cfg_step_lora:
        merge_lora(dit, variant_dir, "cfg_step_lora")
        for sub_cfg in (pipeline.t2v.config, pipeline.continuation.config):
            sub_cfg.cfg_collapse = True
            sub_cfg.num_sampling_steps = 8
            sub_cfg.text_guidance_scale = 0.0
        cfg.cfg_collapse = True
        cfg.num_sampling_steps = 8
        cfg.text_guidance_scale = 0.0
        print(f"  [cfg_step_lora] pipeline flipped to fast mode: "
              f"cfg_collapse=True, 8 steps, guidance_scale=0 "
              f"(applies to both T2V seed + Continuation segments)")

    return pipeline, cfg, variant_dir


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Interactive inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompts-file", type=pathlib.Path,
                     help="Path to a file with one prompt per line")
    grp.add_argument("--prompt", action="append", default=None,
                     help="Per-segment prompt (repeat for multiple segments)")
    parser.add_argument("--negative-prompt", default="",
                        help="Same negative prompt for all segments")
    parser.add_argument("--negative-prompts-file", type=pathlib.Path,
                        help="Per-segment negative prompts file (same line "
                             "count as positive prompts)")
    parser.add_argument("--num-frames-per-segment", type=int, default=93)
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-step-lora", action="store_true",
                        help="Pre-merge cfg_step_lora for the fast path "
                             "(applies to both T2V seed + Continuation segments)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_interactive.mp4"))
    parser.add_argument("--write-segments", action="store_true",
                        help="Write intermediate MP4s per segment")
    args = parser.parse_args()

    # Resolve prompts list
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
    else:
        prompts = list(args.prompt)
    if not prompts:
        parser.error("Must specify at least one prompt")

    # Resolve negative prompts list
    if args.negative_prompts_file:
        neg_prompts = load_prompts(args.negative_prompts_file)
        if len(neg_prompts) != len(prompts):
            parser.error(
                f"--negative-prompts-file has {len(neg_prompts)} entries "
                f"but --prompts has {len(prompts)} entries"
            )
    else:
        neg_prompts = [args.negative_prompt] * len(prompts)

    num_segments = len(prompts)
    expected_total = (args.num_frames_per_segment
                      + (num_segments - 1)
                        * (args.num_frames_per_segment - args.num_cond_frames))
    duration_s = expected_total / 15.0

    print("=== LongCat-Video Interactive MLX inference ===")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:78]}{'...' if len(p) > 78 else ''}")
    print(f"\nSegments: {num_segments}, total {expected_total} frames "
          f"≈ {duration_s:.1f}s @ 15fps")
    print(f"Output  : {args.out}\n")

    print("[1/5] Building pipeline...")
    t0 = time.time()
    pipeline, cfg, variant_dir = build_pipeline(
        args.weights, args.num_frames_per_segment, args.num_cond_frames,
        args.height, args.width,
        with_cfg_step_lora=args.cfg_step_lora,
    )
    if args.num_steps:
        pipeline.t2v.config.num_sampling_steps = args.num_steps
        pipeline.continuation.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print(f"[2/5] Tokenizing + encoding {num_segments} prompts via umT5...")
    prompts_encoded = []
    for p, n in zip(prompts, neg_prompts):
        te, tm, ue, um = encode_prompts(
            pipeline.text_encoder, p, n, variant_dir,
        )
        prompts_encoded.append((te, tm, ue, um))

    print(f"[3/5] Generating {num_segments} segments...")

    def on_done(seg_idx, label, frames):
        elapsed = time.time() - t1
        print(f"  segment {seg_idx + 1}/{num_segments}: "
              f"{label[:50]}{'...' if len(label) > 50 else ''} "
              f"→ {frames.shape[0]} frames ({elapsed:.1f}s elapsed)")
        if args.write_segments:
            seg_path = args.out.with_name(
                f"{args.out.stem}_seg{seg_idx+1:02d}.mp4"
            )
            save_video_mp4(frames, seg_path, fps=cfg.target_fps)

    t1 = time.time()
    all_frames = pipeline(
        prompts_encoded=prompts_encoded,
        seed=args.seed,
        on_segment_done=on_done,
        prompt_labels=prompts,
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
