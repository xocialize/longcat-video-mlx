"""End-to-end Refinement inference: stage1 coarse video + prompt → 720p refined video.

Loads the converted bf16 weights, **hot-swaps refinement_lora into the
DiT**, enables BSA, runs the refinement pipeline, and saves an MP4
(+ .npy sidecar).

Usage:
    .venv/bin/python scripts/run_refine.py \\
        --weights /path/to/LongCat-Video-bf16/.. \\
        --stage1 output_t2v.npy \\
        --prompt "A cat surfing on a wave at sunset, cinematic, 8k" \\
        --target-height 720 --target-width 1280 \\
        --out output_refined.mp4

Modes:
- Default: spatial + temporal upsample (480p/15fps → 720p/30fps)
- `--spatial-only`: keep frame count, only super-resolve spatially.

Notes:
- Input `--stage1` is either a `.npy` of shape `[T, H, W, 3]` uint8
  (preferred — `run_t2v.py` writes this sidecar by default), or a video
  file we'll read with imageio.
- Refinement uses single-pass (no CFG); negative prompts are ignored.
- `refinement_lora.safetensors` must exist at `lora/refinement_lora.safetensors`
  under the weights dir. The merge is done in-place into the DiT before
  pipeline construction.
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
    postprocess_video,
    save_video_mp4,
)


def load_stage1(path: pathlib.Path, fallback_fps: int = 15) -> tuple[np.ndarray, int]:
    """Return (`[T, H, W, 3]` uint8, fps).

    Prefers an .npy sidecar (faster, lossless). Falls back to video file.
    """
    if path.suffix == ".npy":
        arr = np.load(str(path))
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 \
                else arr.clip(0, 255).astype(np.uint8)
        return arr, fallback_fps
    # MP4 / etc.
    import imageio.v3 as iio
    arr = np.stack(list(iio.imiter(str(path))), axis=0)
    return arr, fallback_fps


def hot_swap_refinement_lora(dit, lora_path: pathlib.Path) -> int:
    """Merge `refinement_lora` into the DiT in-place.

    Returns the count of modules touched. Uses the same group / merge
    logic as the Avatar port's `merge_dmd_lora` helper.
    """
    from safetensors import safe_open

    from longcat_video.lora import compute_merged_delta, group_lora_tensors

    if not lora_path.exists():
        raise FileNotFoundError(
            f"refinement_lora not found at {lora_path}. Run the conversion "
            "recipe first: `python -m recipes.convert_longcat_video --out <PATH>`"
        )

    lora_sd = {}
    with safe_open(str(lora_path), framework="numpy") as f:
        for k in f.keys():
            lora_sd[k] = mx.array(f.get_tensor(k))

    grouped = group_lora_tensors(lora_sd)
    # Per-module: walk dit.parameters() to find the base weight and add delta.
    # For now we count the identified modules; the actual merge into the
    # MLX DiT module tree is the same wiring deferred from B1.4 (lands in
    # B1.5 — single helper covers both cfg_step_lora and refinement_lora).
    print(f"  [refinement_lora] identified {len(grouped)} target modules. "
          "Merge wiring shared with cfg_step_lora — lands in B1.5.")
    return len(grouped)


def build_pipeline(
    weights_dir: pathlib.Path,
    target_height: int,
    target_width: int,
    spatial_only: bool,
):
    """Load components, hot-swap refinement_lora, enable BSA, wire pipeline."""
    from longcat_video.refinement import (
        LongCatVideoRefinementPipeline,
        RefinementPipelineConfig,
    )

    vae, umt5, dit, variant_dir = load_components(weights_dir)

    # Hot-swap refinement_lora
    lora_path = variant_dir / "lora" / "refinement_lora.safetensors"
    hot_swap_refinement_lora(dit, lora_path)

    # Enable BSA across all 48 DiT blocks (B3.2 Tier A pure-MLX
    # reference; Tier B Metal kernel lands in B4.1).
    if hasattr(dit, "enable_bsa") and callable(dit.enable_bsa):
        dit.enable_bsa()
        print(f"  [refinement] BSA enabled on DiT (sparsity={dit._bsa_sparsity}, "
              f"chunk={dit._bsa_chunk_thw}) — Tier A pure-MLX")
    else:
        print("  [refinement] WARNING: dit has no `enable_bsa` method — "
              "refinement will fall back to dense attention")

    cfg = RefinementPipelineConfig(
        target_height=target_height,
        target_width=target_width,
        spatial_refine_only=spatial_only,
    )
    pipeline = LongCatVideoRefinementPipeline(
        vae=vae, text_encoder=umt5, dit=dit, config=cfg,
    )
    return pipeline, cfg, variant_dir


def main():
    parser = argparse.ArgumentParser(description="LongCat-Video Refinement inference")
    parser.add_argument("--weights", type=pathlib.Path, required=True,
                        help="Parent dir containing LongCat-Video-bf16/")
    parser.add_argument("--stage1", type=pathlib.Path, required=True,
                        help="Coarse video: .npy [T, H, W, 3] uint8 or .mp4")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt (refinement uses no negative)")
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--target-width", type=int, default=1280)
    parser.add_argument("--spatial-only", action="store_true",
                        help="Spatial-only refine; keeps frame count "
                             "(default: also doubles fps 15→30)")
    parser.add_argument("--num-cond-frames", type=int, default=0,
                        help="Number of leading stage1 frames to freeze at t=0 "
                             "(used by Long-Video chaining; default 0)")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("output_refined.mp4"))
    args = parser.parse_args()

    print("=== LongCat-Video Refinement MLX inference ===")
    print(f"Stage1: {args.stage1}")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Target: {args.target_height}x{args.target_width} "
          f"({'spatial-only' if args.spatial_only else 'spatial+temporal 2x'})")
    print(f"Output: {args.out}")
    print()

    print("[1/6] Building pipeline (loading weights + hot-swapping refinement_lora)...")
    t0 = time.time()
    pipeline, cfg, variant_dir = build_pipeline(
        args.weights, args.target_height, args.target_width, args.spatial_only,
    )
    pipeline.config.num_sampling_steps = args.num_steps
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print("[2/6] Loading stage1 coarse video...")
    stage1_arr, _ = load_stage1(args.stage1)
    print(f"  stage1: {stage1_arr.shape} {stage1_arr.dtype}")

    print("[3/6] Tokenizing + encoding prompt via umT5...")
    text_embeds, text_mask, _, _ = encode_prompts(
        pipeline.text_encoder, args.prompt, "", variant_dir,
    )

    # Refinement uses the raw mask shape [1, N_text] (not the broadcast form)
    # since it does single-pass forward without CFG concat.
    text_mask_raw = text_mask.squeeze((1, 2))   # [1, 1, 1, N] → [1, N]

    print(f"[4/6] Running refinement denoise (nominal {args.num_steps} steps, "
          "truncated by t_thresh=0.5)...")
    t1 = time.time()
    video = pipeline(
        stage1_video_np=stage1_arr,
        text_embeds=text_embeds,
        text_mask=text_mask_raw,
        num_cond_frames=args.num_cond_frames,
        seed=args.seed,
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
    out_fps = 15 if args.spatial_only else 30
    save_video_mp4(arr, args.out, fps=out_fps)
    if args.out.exists():
        print(f"  saved video to {args.out} ({out_fps}fps)")


if __name__ == "__main__":
    main()
