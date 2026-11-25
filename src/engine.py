"""Online search engine: loads every saved artifact once and answers queries.

Used by the API (src/api.py) and the demo (app.py). Modes:
- rerank: candidates from BM25 + dense, scored by the LambdaRank model (default)
- hybrid: reciprocal rank fusion of BM25 and dense
- dense:  embedding search only
- bm25:   keyword search only

Run the offline pipeline first (src.bm25 is not needed; src.dense,
src.hybrid and src.rerank produce the embeddings, fusion weight and model).
"""
from __future__ import annotations

import json
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import DEPTH, FEATURES, CandidateGenerator
from src.hybrid import CONFIG_PATH as HYBRID_CONFIG
from src.hybrid import rrf_fuse_hits
from src.rerank import MODEL_DIR

MODES = ("rerank", "hybrid", "dense", "bm25")
MODEL_PATH = MODEL_DIR / "lambdarank.txt"


class SearchEngine:
    def __init__(self, products: pd.DataFrame, gen: CandidateGenerator, booster: lgb.Booster,
                 w_bm25: float, rrf_k: int):
        self.products = products.set_index("product_id", drop=False)
        self.gen, self.booster = gen, booster
        self.w_bm25, self.rrf_k = w_bm25, rrf_k

    @classmethod
    def load(cls) -> "SearchEngine":
        from src.data import load_products

        missing = [str(p) for p in (HYBRID_CONFIG, MODEL_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing {missing}; run `python -m src.dense`, `python -m src.hybrid` "
                "and `python -m src.rerank` first")
        products = load_products()
        config = json.loads(HYBRID_CONFIG.read_text())
        return cls(products, CandidateGenerator.build(products),
                   lgb.Booster(model_file=str(MODEL_PATH)), config["w_bm25"], config["rrf_k"])

    def search(self, query: str, k: int = 10, mode: str = "rerank"):
        """Return ([(product_id, score), ...] best first, {step: milliseconds})."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        timings: dict[str, float] = {}
        t = perf_counter()

        def lap(step: str) -> None:
            nonlocal t
            now = perf_counter()
            timings[step] = (now - t) * 1000
            t = now

        if mode == "bm25":
            hits = self.gen.bm25.search(query, k)
            lap("bm25")
        else:
            qvec = self.gen.encode_queries([query])[0]
            lap("encode_query")
            if mode == "dense":
                hits = self.gen.dense.search_vectors(qvec[None, :], k)[0]
                lap("dense")
            elif mode == "hybrid":
                bm25 = self.gen.bm25.search(query, DEPTH)
                dense = self.gen.dense.search_vectors(qvec[None, :], DEPTH)[0]
                lap("retrieve")
                hits = rrf_fuse_hits(bm25, dense, self.w_bm25, self.rrf_k, depth=k)
                lap("fuse")
            else:
                cands = self.gen.candidates(query, qvec)
                lap("retrieve_and_features")
                scores = self.booster.predict(cands[FEATURES])
                pids = cands["product_id"].to_numpy()
                order = np.lexsort((pids, -scores))[:k]
                hits = [(int(pids[i]), float(scores[i])) for i in order]
                lap("rerank_model")
        timings["total"] = sum(timings.values())
        return hits, timings

    def describe(self, hits: list[tuple[int, float]]) -> list[dict]:
        """Product details for a result list."""
        out = []
        for rank, (pid, score) in enumerate(hits, start=1):
            p = self.products.loc[pid]
            rating = p["average_rating"]
            out.append({
                "rank": rank,
                "product_id": int(pid),
                "name": p["product_name"],
                "product_class": p["product_class"],
                "score": round(float(score), 6),
                "average_rating": None if pd.isna(rating) else float(rating),
                "review_count": int(p["review_count"]),
            })
        return out
