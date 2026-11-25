"""Dense (embedding) retrieval with Sentence-Transformers and FAISS.

Each product is turned into a 384-number vector by all-MiniLM-L6-v2, a small
model trained so that texts with similar meaning get nearby vectors. A query
is embedded the same way, and the products whose vectors point in the most
similar direction (highest cosine similarity) are returned. Unlike BM25 this
can match "couch" to "sofa" without a shared word.

- Vectors are L2-normalised, so inner product == cosine similarity.
- FAISS IndexFlatIP does exact search. With 43k products that takes a few
  milliseconds; approximate indexes (HNSW, IVF) only pay off at millions.
- Product embeddings are computed once and cached in artifacts/embeddings/.

Run `python -m src.dense` to embed products (first run only), retrieve the
top 100 for every query and score it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import pandas as pd

from src.evaluate import ARTIFACTS_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PASSAGE_FORMAT = "name. class. description (v1)"  # bump if product_passages changes
EMB_DIR = ARTIFACTS_DIR / "embeddings"


def product_passages(products: pd.DataFrame) -> list[str]:
    """Name first, then class, then description.

    The model reads at most 256 word pieces, so the name and class always fit
    and the description fills whatever room is left.
    """
    passages = []
    for name, cls, desc in zip(products["product_name"], products["product_class"],
                               products["product_description"]):
        parts = [p.strip() for p in (name, cls, desc) if isinstance(p, str) and p.strip()]
        passages.append(". ".join(parts))
    return passages


def load_model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=device)


def encode(encoder, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    vecs = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=progress)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def _cache_key(products: pd.DataFrame) -> dict:
    ids = products["product_id"].to_numpy().astype(np.int64).tobytes()
    return {"model": MODEL_NAME, "passage_format": PASSAGE_FORMAT,
            "n_products": len(products), "ids_sha1": hashlib.sha1(ids).hexdigest()}


def load_or_build_embeddings(products: pd.DataFrame, encoder=None, cache_dir: Path = EMB_DIR) -> np.ndarray:
    """Product embeddings in the same row order as `products`, cached on disk."""
    vec_path, meta_path = cache_dir / "products.npy", cache_dir / "products.json"
    key = _cache_key(products)
    if vec_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == key:
        return np.load(vec_path)

    encoder = encoder or load_model()
    t0 = perf_counter()
    vecs = encode(encoder, product_passages(products), progress=True)
    print(f"embedded {len(vecs):,} products in {perf_counter() - t0:.0f}s")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps(key, indent=2))
    return vecs


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, product_ids, encoder):
        self.encoder = encoder
        self.product_ids = np.asarray(product_ids)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def search_vectors(self, query_vecs: np.ndarray, k: int = 100) -> list[list[tuple[int, float]]]:
        scores, rows = self.index.search(query_vecs, k)
        return [
            [(int(self.product_ids[r]), float(s)) for r, s in zip(row, score) if r >= 0]
            for row, score in zip(rows, scores)
        ]

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[int, float]]]:
        return self.search_vectors(encode(self.encoder, queries), k)

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        return self.search_batch([query], k)[0]


def main(k: int = 100) -> None:
    from src.data import load_products, load_queries
    from src.evaluate import RUN_COLUMNS, evaluate_stage, print_summary, save_run

    products, queries = load_products(), load_queries()
    model = load_model()
    embeddings = load_or_build_embeddings(products, model)
    index = DenseIndex(embeddings, products["product_id"], model)

    t0 = perf_counter()
    results = index.search_batch(queries["query"].tolist(), k)
    print(f"searched {len(queries)} queries in {perf_counter() - t0:.1f}s (batched)")

    rows = [(int(qid), pid, rank, score)
            for qid, hits in zip(queries["query_id"], results)
            for rank, (pid, score) in enumerate(hits, start=1)]
    run = pd.DataFrame(rows, columns=RUN_COLUMNS)
    save_run(run, "dense")
    print_summary("Dense", evaluate_stage("dense", run))


if __name__ == "__main__":
    main()
