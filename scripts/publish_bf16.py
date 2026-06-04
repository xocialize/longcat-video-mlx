"""Publish the converted bf16 LongCat-Video weights to mlx-community on HF.

Uses the same `hf repos create --exist-ok` + `hf upload` pattern from
the Avatar port (skill lesson L19). Resumable on transient failures —
HF's chunked upload picks up where it left off.

Expected layout under `--weights-dir/LongCat-Video-bf16/`:

    vae/                                       (242 MB)
    text_encoder/                              (~11 GB sharded)
    dit/                                       (~26 GB sharded)
    lora/cfg_step_lora.safetensors             (~2.3 GB)
    lora/refinement_lora.safetensors           (~3.0 GB)
    tokenizer/                                 (~20 MB SentencePiece)
    scheduler/                                 (~1 KB)
    pipeline_config.json                       (~1 KB)

Total: ~42 GB.

Usage:
    .venv/bin/python scripts/publish_bf16.py \\
        --weights-dir /Users/dustinnielson/DEV_INT/longcat-video-mlx-weights \\
        --repo mlx-community/LongCat-Video-bf16

The model card at `docs/model-cards/bf16.md` is uploaded as `README.md`.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def ensure_repo(repo_id: str) -> None:
    """Idempotent: `hf repos create --exist-ok`."""
    run(["hf", "repos", "create", repo_id, "--repo-type=model", "--exist-ok"])


def stage_readme(weights_dir: pathlib.Path, model_card: pathlib.Path) -> None:
    """Copy the model card into the variant dir as README.md so HF picks
    it up as the canonical model page on upload.
    """
    dest = weights_dir / "LongCat-Video-bf16" / "README.md"
    print(f"  Staging {model_card} → {dest}")
    shutil.copy2(str(model_card), str(dest))


def upload(repo_id: str, variant_dir: pathlib.Path) -> None:
    """Upload everything under variant_dir to the repo's root."""
    run([
        "hf", "upload",
        repo_id,
        str(variant_dir),
        ".",                          # remote subdir = repo root
        "--repo-type=model",
    ])


def main():
    parser = argparse.ArgumentParser(description="Publish LongCat-Video-bf16 to mlx-community")
    parser.add_argument(
        "--weights-dir",
        type=pathlib.Path,
        required=True,
        help="Parent of LongCat-Video-bf16/ (the conversion output dir)",
    )
    parser.add_argument(
        "--repo",
        default="mlx-community/LongCat-Video-bf16",
        help="HF repo id to publish to",
    )
    parser.add_argument(
        "--model-card",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
            / "docs" / "model-cards" / "bf16.md",
        help="Model card markdown (default: docs/model-cards/bf16.md)",
    )
    parser.add_argument(
        "--skip-stage", action="store_true",
        help="Skip copying the model card into the variant dir (use if "
             "README.md is already current)",
    )
    args = parser.parse_args()

    variant_dir = args.weights_dir / "LongCat-Video-bf16"
    if not variant_dir.exists():
        print(f"ERROR: {variant_dir} does not exist. Run the conversion recipe first.")
        sys.exit(1)

    print(f"=== Publishing {variant_dir} → {args.repo} ===\n")

    print("[1/3] Ensuring repo exists...")
    ensure_repo(args.repo)

    if not args.skip_stage:
        if not args.model_card.exists():
            print(f"ERROR: model card not found at {args.model_card}")
            sys.exit(1)
        print(f"[2/3] Staging model card as README.md...")
        stage_readme(args.weights_dir, args.model_card)
    else:
        print("[2/3] (skipped — using existing README.md)")

    print(f"[3/3] Uploading {variant_dir} → {args.repo}...")
    upload(args.repo, variant_dir)

    print("\nDone. Verify at https://huggingface.co/" + args.repo)


if __name__ == "__main__":
    main()
