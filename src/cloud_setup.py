"""Get data and trained artifacts onto a fresh machine (e.g. Streamlit Community Cloud).

The git repository holds no data and no trained files: the pipeline builds
them (about 20 minutes, mostly embedding the catalogue). A fresh deployment
therefore calls `ensure_ready()` before loading the engine. It downloads WANDS
(pinned commit, SHA-256 checked) and the trained artifacts from a pinned
revision of a public Hugging Face model repo. Where the files already exist,
as on a machine that ran the pipeline, it does nothing.

The artifacts are published with `python scripts/publish_artifacts.py`.
"""
from __future__ import annotations

from pathlib import Path

from src.data import DATA_DIR
from src.download import SHA256, download
from src.evaluate import ARTIFACTS_DIR

ARTIFACT_REPO = "Himanshuy18/hybrid-search-artifacts"
# Pinned upload, so a later re-upload can't silently change a running deployment.
ARTIFACT_REVISION = "97f184aa16acecd0c5fd3b4acc216ff54f89bec6"
ARTIFACT_FILES = [
    "embeddings/products.npy",
    "embeddings/products.json",
    "models/lambdarank.txt",
    "models/lambdarank.json",
    "hybrid_config.json",
    "results.csv",
]


def missing_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    return [name for name in ARTIFACT_FILES if not (artifacts_dir / name).exists()]


def ensure_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> bool:
    """Download any missing artifact. Returns True if something was downloaded."""
    missing = missing_artifacts(artifacts_dir)
    if not missing:
        return False
    from huggingface_hub import hf_hub_download

    for name in missing:
        hf_hub_download(ARTIFACT_REPO, filename=name, revision=ARTIFACT_REVISION,
                        local_dir=artifacts_dir)
    return True


def ensure_data(data_dir: Path = DATA_DIR) -> bool:
    """Download WANDS if any file is missing. Returns True if it downloaded."""
    if all((data_dir / name).exists() for name in SHA256):
        return False
    download(data_dir)
    return True


def ensure_ready() -> None:
    ensure_data()
    ensure_artifacts()
