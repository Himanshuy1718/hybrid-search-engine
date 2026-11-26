"""Publish the Streamlit demo as a Hugging Face Docker Space.

    hf auth login                                            # once, with a write token
    python scripts/deploy_hf_space.py --space <user>/hybrid-product-search
    python scripts/deploy_hf_space.py --stage-only <folder>  # just assemble the files

Uploads the code, the Space Dockerfile and card (deploy/huggingface/), and the
small trained artifacts: product embeddings (~63 MB), the LambdaRank model and
the fusion weight. Hugging Face then builds the image, downloading WANDS and
the MiniLM model itself. Run the offline pipeline first so artifacts/ exists.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cloud_setup import ARTIFACT_FILES as ARTIFACTS  # noqa: E402

SPACE_FILES = ROOT / "deploy" / "huggingface"


def stage(dest: Path) -> Path:
    """Copy everything the Space needs into dest."""
    missing = [a for a in ARTIFACTS if not (ROOT / "artifacts" / a).exists()]
    if missing:
        raise SystemExit(f"missing artifacts {missing}: run the pipeline first (see README)")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SPACE_FILES / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(SPACE_FILES / "README.md", dest / "README.md")
    shutil.copy2(ROOT / "requirements.txt", dest / "requirements.txt")
    shutil.copy2(ROOT / "app.py", dest / "app.py")
    shutil.copytree(ROOT / "src", dest / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for a in ARTIFACTS:
        target = dest / "artifacts" / a
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "artifacts" / a, target)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--space", help="Space id, e.g. username/hybrid-product-search")
    group.add_argument("--stage-only", type=Path, help="assemble the files into this folder and stop")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    if args.stage_only:
        print(f"staged in {stage(args.stage_only)}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        folder = stage(Path(tmp))
        api.create_repo(args.space, repo_type="space", space_sdk="docker",
                        private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                          commit_message="Deploy hybrid product search demo")
    print(f"uploaded; the Space is building: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
