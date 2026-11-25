"""Candidate generation and features for the re-ranker.

Candidates for a query are the union of BM25's top 100 and dense's top 100
(up to 200 products). Each candidate gets the features below. The same code
runs offline for training and online in the API, so the model sees exactly
the same features in both places.

Leakage note: query.csv has a `query_class` column. It matches the class of
64% of Exact products and 4% of Irrelevant ones, but it describes the judged
products and would not exist for a new query typed into a search box. Using
it would inflate offline scores with information the live system can't
have, so it is never used. "Category match" is instead computed from what is
available at query time (class_overlap, class_vote).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.bm25 import BM25Index, tokenize
from src.dense import DenseIndex, encode, load_model, load_or_build_embeddings

FEATURES = [
    "bm25_score",       # BM25 over name + class + description
    "bm25_rank",        # rank of that score among all products (1 = best)
    "bm25_name_score",  # BM25 over the product name only
    "cosine",           # embedding cosine similarity
    "dense_rank",       # rank of that similarity among all products
    "name_overlap",     # share of query words found in the product name
    "name_phrase",      # 1 if the whole query appears, in order, in the name
    "class_overlap",    # share of query words found in the product class
    "class_vote",       # share of the first-stage top results with this product's class
    "average_rating",   # missing when the product has no ratings
    "review_count",
    "query_len",        # number of query words (lets the model treat short/long queries differently)
]
DEPTH = 100     # candidates taken from each retriever
VOTE_TOP = 10   # top results per retriever used for class_vote


def _singular(t: str) -> str:
    """Crude plural stripping so 'tables' matches 'table' in the overlap features."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("ches", "shes", "xes", "sses")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def norm_terms(text: str) -> list[str]:
    return [_singular(t) for t in tokenize(text)]


def rank_of(full: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """1 + number of products scoring strictly higher. Ties share a rank."""
    ascending_neg = -np.sort(full)[::-1]
    return np.searchsorted(ascending_neg, -full[rows], side="left") + 1


class CandidateGenerator:
    def __init__(self, products: pd.DataFrame, bm25: BM25Index, bm25_name: BM25Index,
                 dense: DenseIndex, embeddings: np.ndarray):
        self.bm25, self.bm25_name, self.dense = bm25, bm25_name, dense
        self.embeddings = embeddings
        self.product_ids = products["product_id"].to_numpy()
        self.row_of = bm25.row_of
        self.name_terms = [frozenset(norm_terms(n)) for n in products["product_name"]]
        self.name_text = [f" {' '.join(tokenize(n))} " for n in products["product_name"]]
        self.class_terms = [frozenset(norm_terms(c)) for c in products["product_class"]]
        self.product_class = products["product_class"].to_numpy()
        self.rating = products["average_rating"].to_numpy(dtype=float)
        self.review_count = products["review_count"].to_numpy(dtype=float)

    @classmethod
    def build(cls, products: pd.DataFrame, encoder=None) -> "CandidateGenerator":
        encoder = encoder or load_model()
        embeddings = load_or_build_embeddings(products, encoder)
        return cls(products, BM25Index(products), BM25Index(products, fields=("product_name",)),
                   DenseIndex(embeddings, products["product_id"], encoder), embeddings)

    def encode_queries(self, queries) -> np.ndarray:
        return encode(self.dense.encoder, list(queries))

    def candidates(self, query: str, query_vec: np.ndarray, depth: int = DEPTH) -> pd.DataFrame:
        """One row per candidate product: product_id plus FEATURES."""
        bm25_full = self.bm25.scores(query)
        cos_full = self.embeddings @ query_vec
        bm25_top = self.bm25.top_rows(bm25_full, depth)
        dense_hits = self.dense.search_vectors(query_vec[None, :].astype(np.float32), depth)[0]
        dense_top = np.array([self.row_of[pid] for pid, _ in dense_hits], dtype=int)
        rows = np.union1d(bm25_top, dense_top).astype(int)

        q_terms = set(norm_terms(query))
        n_q = len(q_terms)
        q_phrase = f" {' '.join(tokenize(query))} "
        slots = [self.product_class[r] for r in [*bm25_top[:VOTE_TOP], *dense_top[:VOTE_TOP]]]
        votes = Counter(c for c in slots if c)
        n_slots = max(len(slots), 1)

        return pd.DataFrame({
            "product_id": self.product_ids[rows],
            "bm25_score": bm25_full[rows],
            "bm25_rank": rank_of(bm25_full, rows),
            "bm25_name_score": self.bm25_name.scores(query)[rows],
            "cosine": cos_full[rows],
            "dense_rank": rank_of(cos_full, rows),
            "name_overlap": [len(q_terms & self.name_terms[r]) / n_q if n_q else 0.0 for r in rows],
            "name_phrase": [float(n_q > 0 and q_phrase in self.name_text[r]) for r in rows],
            "class_overlap": [len(q_terms & self.class_terms[r]) / n_q if n_q else 0.0 for r in rows],
            "class_vote": [votes.get(self.product_class[r], 0) / n_slots for r in rows],
            "average_rating": self.rating[rows],
            "review_count": self.review_count[rows],
            "query_len": float(len(tokenize(query))),
        })


def build_feature_table(gen: CandidateGenerator, queries: pd.DataFrame,
                        qrels: dict[int, dict[int, int]]) -> pd.DataFrame:
    """Candidates and features for every query, with the judged gain (0 if unlabelled)."""
    query_vecs = gen.encode_queries(queries["query"])
    frames = []
    for qid, text, qvec in zip(queries["query_id"], queries["query"], query_vecs):
        f = gen.candidates(text, qvec)
        judged = qrels.get(int(qid), {})
        f.insert(0, "query_id", int(qid))
        f["gain"] = [judged.get(int(pid), 0) for pid in f["product_id"]]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
