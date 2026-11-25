import pandas as pd
import pytest

from src.hybrid import cross_validated_hybrid, pick_weight, rrf_fuse


def run_of(query_id, product_ids):
    return pd.DataFrame({"query_id": query_id, "product_id": product_ids,
                         "rank": range(1, len(product_ids) + 1), "score": 0.0})


# BM25 ranks: A=1 (rank 1), B=2 (rank 2).  Dense ranks: B=2 (rank 1), C=3 (rank 2).
BM25 = run_of(0, [1, 2])
DENSE = run_of(0, [2, 3])


def test_rrf_by_hand_equal_weights():
    fused = rrf_fuse(BM25, DENSE, w_bm25=0.5, k=60)
    scores = dict(zip(fused["product_id"], fused["score"]))
    assert scores[1] == pytest.approx(0.5 / 61)
    assert scores[2] == pytest.approx(0.5 / 62 + 0.5 / 61)  # in both lists -> wins
    assert scores[3] == pytest.approx(0.5 / 62)
    assert fused["product_id"].tolist() == [2, 1, 3]
    assert fused["rank"].tolist() == [1, 2, 3]


def test_weight_one_is_pure_bm25_and_zero_is_pure_dense():
    assert rrf_fuse(BM25, DENSE, w_bm25=1.0)["product_id"].tolist() == [1, 2]
    assert rrf_fuse(BM25, DENSE, w_bm25=0.0)["product_id"].tolist() == [2, 3]


def test_depth_cuts_each_query():
    fused = rrf_fuse(BM25, DENSE, w_bm25=0.5, depth=2)
    assert fused["product_id"].tolist() == [2, 1]


def test_pick_weight_uses_training_queries_only():
    # Query 3 (the test query) would prefer weight 0.0, the training queries 1.0.
    ndcg = pd.DataFrame({0.0: [0.1, 0.1, 1.0], 1.0: [0.9, 0.9, 0.0]}, index=[1, 2, 3])
    assert pick_weight(ndcg, train_ids=[1, 2]) == 1.0
    assert pick_weight(ndcg, train_ids=[3]) == 0.0


def test_cross_validation_never_tunes_on_the_test_fold():
    # Two queries in two folds. Query 0 is best served by BM25, query 1 by dense.
    # Each fold must take the weight its *other* fold prefers.
    bm25 = pd.concat([run_of(0, [10, 11]), run_of(1, [21, 20])])
    dense = pd.concat([run_of(0, [11, 10]), run_of(1, [20, 21])])
    qrels = {0: {10: 2}, 1: {20: 2}}
    folds = {0: 0, 1: 1}
    _, chosen, _ = cross_validated_hybrid(bm25, dense, qrels, [0, 1], folds, weights=[0.0, 1.0])
    assert chosen == {0: 0.0, 1: 1.0}
