"""Metric checks against small examples worked out by hand."""
import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from src.metrics import dcg, evaluate_run, ndcg_at_k, recall_at_k, reciprocal_rank

# One query: a and d are Exact (2), b is Partial (1), c is Irrelevant (0).
# "x" and "y" have no label, so they count as gain 0.
JUDGED = {"a": 2, "b": 1, "c": 0, "d": 2}
LOG2_3 = math.log2(3)  # discount at position 2


# ---------- DCG / NDCG ----------

def test_dcg_by_hand():
    # positions 1..3 -> discounts log2(2)=1, log2(3), log2(4)=2
    assert dcg([2, 1, 2], k=3) == pytest.approx(2 / 1 + 1 / LOG2_3 + 2 / 2)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "d", "b", "c"], JUDGED, k=10) == pytest.approx(1.0)


def test_ndcg_by_hand():
    # ranking gains: c=0, a=2, b=1  -> DCG  = 0/1 + 2/log2(3) + 1/2
    # ideal gains:   2, 2, 1        -> IDCG = 2/1 + 2/log2(3) + 1/2
    got = ndcg_at_k(["c", "a", "b", "x"], JUDGED, k=3)
    expected = (2 / LOG2_3 + 1 / 2) / (2 + 2 / LOG2_3 + 1 / 2)
    assert got == pytest.approx(expected)  # about 0.4684


def test_ndcg_unlabelled_products_count_as_zero():
    assert ndcg_at_k(["x", "y"], JUDGED, k=10) == 0.0


def test_ndcg_ideal_uses_all_judged_products_not_just_retrieved():
    # Only "b" (Partial) was retrieved. If the ideal were built from the
    # retrieved items alone this would wrongly score 1.0.
    got = ndcg_at_k(["b"], JUDGED, k=2)
    expected = (1 / 1) / (2 / 1 + 2 / LOG2_3)
    assert got == pytest.approx(expected)


def test_ndcg_short_ranking_shorter_than_k():
    got = ndcg_at_k(["a"], JUDGED, k=10)
    ideal = 2 + 2 / LOG2_3 + 1 / 2  # all three positive gains fit in k=10
    assert got == pytest.approx(2 / ideal)


def test_ndcg_matches_sklearn_when_all_judged_items_are_ranked():
    # sklearn also uses linear gain and log2 discount, but builds the ideal
    # from the items you pass. With every judged item present they agree.
    ranking = ["c", "b", "d", "a"]
    scores = [4, 3, 2, 1]  # descending -> same order as `ranking`
    true = [JUDGED[p] for p in ranking]
    expected = ndcg_score([true], [scores], k=3)
    assert ndcg_at_k(ranking, JUDGED, k=3) == pytest.approx(expected)


def test_ndcg_raises_when_query_has_no_relevant_products():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 0, "b": 0}, k=10)


# ---------- MRR ----------

def test_rr_first_exact_at_position_three():
    # b is Partial and does not count as relevant
    assert reciprocal_rank(["c", "b", "a"], JUDGED, k=10) == pytest.approx(1 / 3)


def test_rr_first_position():
    assert reciprocal_rank(["d", "a"], JUDGED, k=10) == 1.0


def test_rr_exact_beyond_cutoff_is_zero():
    assert reciprocal_rank(["c", "b", "a"], JUDGED, k=2) == 0.0


# ---------- Recall ----------

def test_recall_by_hand():
    ranking = ["a", "x", "d"]  # Exact products are {a, d}
    assert recall_at_k(ranking, JUDGED, k=2) == pytest.approx(1 / 2)
    assert recall_at_k(ranking, JUDGED, k=3) == pytest.approx(2 / 2)


def test_recall_ignores_partial():
    assert recall_at_k(["b"], JUDGED, k=10) == 0.0


def test_recall_raises_without_exact_products():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 1}, k=10)


# ---------- shared checks ----------

@pytest.mark.parametrize("metric", [ndcg_at_k, reciprocal_rank, recall_at_k])
def test_duplicates_are_rejected(metric):
    with pytest.raises(ValueError):
        metric(["a", "a"], JUDGED, k=10)


def test_evaluate_run_scores_missing_query_as_zero():
    qrels = {1: JUDGED, 2: {"z": 2}}
    run = {1: ["a", "d", "b"]}  # query 2 missing
    table = evaluate_run(run, qrels, query_ids=[1, 2])
    assert list(table.index) == [1, 2]
    assert table.loc[1, "mrr@10"] == 1.0
    assert table.loc[2].tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(table.to_numpy()).all()
