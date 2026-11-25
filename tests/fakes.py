"""Tiny catalogue and fake encoder shared by tests (no model download)."""
import numpy as np
import pandas as pd

from src.bm25 import BM25Index
from src.dense import DenseIndex
from src.features import CandidateGenerator

WORDS = {"coffee": [1, 0, 0, 0], "table": [0, 1, 0, 0], "sofa": [0, 0, 1, 0],
         "couch": [0, 0, 0.9, 0.1], "mug": [0, 0, 0, 1]}


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True, **_):
        vecs = np.array([np.sum([WORDS.get(w, [0, 0, 0, 0]) for w in t.lower().split()], axis=0)
                         for t in texts], dtype=np.float32) + 1e-3
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


PRODUCTS = pd.DataFrame({
    "product_id": [0, 1, 2, 3, 4],
    "product_name": ["oak coffee table", "glass coffee tables", "red sofa", "blue couch", "coffee mug"],
    "product_class": ["Coffee Tables", "Coffee Tables", "Sofas", "Sofas", ""],
    "product_description": [""] * 5,
    "average_rating": [4.5, np.nan, 4.0, 3.0, 5.0],
    "review_count": [10, 0, 3, 1, 99],
})


def make_gen() -> CandidateGenerator:
    enc = FakeEncoder()
    emb = enc.encode(list(PRODUCTS["product_name"]))
    return CandidateGenerator(PRODUCTS, BM25Index(PRODUCTS), BM25Index(PRODUCTS, fields=("product_name",)),
                              DenseIndex(emb, PRODUCTS["product_id"], enc), emb)
