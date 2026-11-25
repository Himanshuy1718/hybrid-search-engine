"""Ranking metrics: NDCG@k, MRR@k and Recall@k.

Conventions (also stated in the README):
- Gains: Exact = 2, Partial = 1, Irrelevant = 0.
- A product with no label for a query has gain 0.
- "Relevant" for MRR and Recall means Exact (gain 2). Partial does not count.
- DCG uses linear gain with a log2 discount: sum(gain_i / log2(i + 1)), i = 1..k.
- The ideal DCG is built from *all* labelled products for the query, not just
  the ones a retriever happened to return. Otherwise a retriever that misses
  every good product could still score well.

Every stage (BM25, dense, hybrid, re-ranker) is scored with `evaluate_run`
on the same list of query ids, so the numbers are directly comparable.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence

import pandas as pd

EXACT_GAIN = 2

Qrels = Mapping[Hashable, Mapping[Hashable, int]]  # query_id -> {product_id: gain}
Run = Mapping[Hashable, Sequence[Hashable]]  # query_id -> ranked product ids


def _check_no_duplicates(ranking: Sequence[Hashable]) -> None:
    # A duplicate would be counted twice and inflate every metric.
    # It almost always means a bug in candidate merging, so fail loudly.
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate product ids")


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of the first k gains."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranking: Sequence[Hashable], judged: Mapping[Hashable, int], k: int = 10) -> float:
    """NDCG@k for one query.

    ranking: product ids in ranked order (best first).
    judged:  {product_id: gain} for this query. Missing ids have gain 0.
    """
    _check_no_duplicates(ranking)
    ideal = sorted(judged.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        raise ValueError("query has no relevant products; exclude it from evaluation")
    gains = [judged.get(pid, 0) for pid in ranking[:k]]
    return dcg(gains, k) / idcg


def reciprocal_rank(ranking: Sequence[Hashable], judged: Mapping[Hashable, int], k: int = 10) -> float:
    """1 / position of the first Exact product within the top k, else 0."""
    _check_no_duplicates(ranking)
    for position, pid in enumerate(ranking[:k], start=1):
        if judged.get(pid, 0) >= EXACT_GAIN:
            return 1.0 / position
    return 0.0


def recall_at_k(ranking: Sequence[Hashable], judged: Mapping[Hashable, int], k: int = 100) -> float:
    """Share of the query's Exact products that appear in the top k."""
    _check_no_duplicates(ranking)
    relevant = {pid for pid, gain in judged.items() if gain >= EXACT_GAIN}
    if not relevant:
        raise ValueError("query has no Exact products; exclude it from evaluation")
    hits = sum(1 for pid in ranking[:k] if pid in relevant)
    return hits / len(relevant)


def evaluate_run(
    run: Run,
    qrels: Qrels,
    query_ids: Iterable[Hashable],
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 100,
) -> pd.DataFrame:
    """Score a run query by query.

    query_ids is required so every stage is scored on exactly the same
    queries. A query missing from the run counts as an empty ranking
    (all metrics 0); it is never silently skipped.
    """
    rows = []
    for qid in query_ids:
        ranking = list(run.get(qid, []))
        judged = qrels[qid]
        rows.append(
            {
                "query_id": qid,
                f"ndcg@{ndcg_k}": ndcg_at_k(ranking, judged, ndcg_k),
                f"mrr@{mrr_k}": reciprocal_rank(ranking, judged, mrr_k),
                f"recall@{recall_k}": recall_at_k(ranking, judged, recall_k),
            }
        )
    return pd.DataFrame(rows).set_index("query_id")
