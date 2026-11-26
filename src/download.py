"""Download the WANDS dataset into data/ and verify it.

    python -m src.download

Files come from a pinned commit of github.com/wayfair/WANDS (the latest, from
January 2022) and are checked against SHA-256 hashes, so every run of the
pipeline, locally or in Docker, starts from byte-identical data.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from src.data import DATA_DIR

WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
URL = "https://raw.githubusercontent.com/wayfair/WANDS/{commit}/dataset/{name}"
SHA256 = {
    "product.csv": "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6",
    "query.csv": "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95",
    "label.csv": "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        path = data_dir / name
        if path.exists() and sha256(path) == expected:
            print(f"{name}: already present")
            continue
        tmp = path.with_suffix(".part")
        print(f"{name}: downloading")
        urllib.request.urlretrieve(URL.format(commit=WANDS_COMMIT, name=name), tmp)
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink()
            raise RuntimeError(f"{name}: checksum mismatch ({actual} != {expected})")
        tmp.replace(path)
        print(f"{name}: ok")


if __name__ == "__main__":
    download()
