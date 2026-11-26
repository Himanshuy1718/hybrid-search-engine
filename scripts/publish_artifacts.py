"""Upload the trained artifacts to a public Hugging Face model repo.

    hf auth login                         # once, with a write token
    python scripts/publish_artifacts.py   # prints the commit to pin in src/cloud_setup.py

A deployment that has only the git repository downloads these files with
`src.cloud_setup.ensure_ready()` instead of re-running the pipeline.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cloud_setup import ARTIFACT_FILES, ARTIFACT_REPO, missing_artifacts  # noqa: E402
from src.evaluate import ARTIFACTS_DIR  # noqa: E402

CARD = """---
tags:
  - information-retrieval
  - learning-to-rank
  - product-search
---

# Hybrid search artifacts (WANDS)

Trained files for https://github.com/Himanshuy1718/hybrid-search-engine:

- `embeddings/products.npy`: all-MiniLM-L6-v2 embeddings of the 42,994 WANDS
  products (float32, L2-normalised, row order = product_id order)
- `models/lambdarank.txt`: LightGBM LambdaRank re-ranker (174 trees)
- `hybrid_config.json`: reciprocal rank fusion weight
- `results.csv`: cross-validated results of every stage

They are produced by the pipeline in that repository; see its README for how
they were built and evaluated.
"""


def main() -> None:
    missing = missing_artifacts()
    if missing:
        raise SystemExit(f"missing artifacts {missing}: run the pipeline first")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(ARTIFACT_REPO, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(CARD, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=ARTIFACT_REPO)
    info = api.upload_folder(folder_path=str(ARTIFACTS_DIR), repo_id=ARTIFACT_REPO,
                             allow_patterns=ARTIFACT_FILES,
                             commit_message="Upload trained artifacts")
    print(f"https://huggingface.co/{ARTIFACT_REPO}")
    print(f"revision: {info.oid}")


if __name__ == "__main__":
    main()
