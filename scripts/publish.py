"""Publish a converted LongCat-Video variant to mlx-community on HF.

Variant-aware version of the original `publish_bf16.py`. Picks the right
variant directory + matching model card based on `--variant`.

Uses the `hf repos create --exist-ok` + `hf upload` pattern (skill lesson
L19; L28 for the v1.17 CLI command names). Resumable on transient failures
— HF's chunked upload picks up where it left off.

Variant layouts:

    LongCat-Video-bf16/   (~42 GB; the reference variant)
    LongCat-Video-q4/     (~25 GB; 4-bit DiT)
    LongCat-Video-q8/     (~31 GB; 8-bit DiT)

Usage:
    .venv/bin/python scripts/publish.py \\
        --weights-dir /Users/dustinnielson/DEV_INT/longcat-video-mlx-weights \\
        --variant q4

The model card at `docs/model-cards/{variant}.md` is staged as `README.md`
in the variant dir before uploading.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


VARIANT_REPO: dict[str, str] = {
    "bf16": "mlx-community/LongCat-Video-bf16",
    "q4":   "mlx-community/LongCat-Video-q4",
    "q8":   "mlx-community/LongCat-Video-q8",
}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def ensure_repo(repo_id: str) -> None:
    """Idempotent: `hf repos create --exist-ok`."""
    run(["hf", "repos", "create", repo_id, "--repo-type=model", "--exist-ok"])


def stage_readme(variant_dir: pathlib.Path, model_card: pathlib.Path) -> None:
    """Copy the model card into the variant dir as README.md so HF picks
    it up as the canonical model page on upload.
    """
    dest = variant_dir / "README.md"
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
    parser = argparse.ArgumentParser(description="Publish a LongCat-Video variant to mlx-community")
    parser.add_argument(
        "--weights-dir",
        type=pathlib.Path,
        required=True,
        help="Parent of LongCat-Video-{bf16,q4,q8}/ (the conversion output dir)",
    )
    parser.add_argument(
        "--variant", required=True,
        choices=["bf16", "q4", "q8"],
        help="Which variant to publish",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="HF repo id (default: mlx-community/LongCat-Video-{variant})",
    )
    parser.add_argument(
        "--model-card",
        type=pathlib.Path,
        default=None,
        help="Model card markdown (default: docs/model-cards/{variant}.md)",
    )
    parser.add_argument(
        "--skip-stage", action="store_true",
        help="Skip copying the model card into the variant dir",
    )
    args = parser.parse_args()

    repo = args.repo or VARIANT_REPO[args.variant]
    model_card = args.model_card or (
        pathlib.Path(__file__).parent.parent
        / "docs" / "model-cards" / f"{args.variant}.md"
    )
    variant_dir = args.weights_dir / f"LongCat-Video-{args.variant}"

    if not variant_dir.exists():
        print(f"ERROR: {variant_dir} does not exist. Run the conversion recipe first.")
        sys.exit(1)

    print(f"=== Publishing {variant_dir} → {repo} ===\n")

    print("[1/3] Ensuring repo exists...")
    ensure_repo(repo)

    if not args.skip_stage:
        if not model_card.exists():
            print(f"ERROR: model card not found at {model_card}")
            sys.exit(1)
        print(f"[2/3] Staging model card as README.md...")
        stage_readme(variant_dir, model_card)
    else:
        print("[2/3] (skipped — using existing README.md)")

    print(f"[3/3] Uploading {variant_dir} → {repo}...")
    upload(repo, variant_dir)

    print(f"\nDone. Verify at https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
