"""Feature checks on a tiny catalogue with a fake encoder (no model download)."""
import numpy as np
import pytest

from src.features import FEATURES, _singular, rank_of
from tests.fakes import make_gen


def features_for(query, depth=100):
    gen = make_gen()
    f = gen.candidates(query, gen.encode_queries([query])[0], depth=depth)
    return f.set_index("product_id")


@pytest.mark.parametrize("word, expected", [
    ("tables", "table"), ("benches", "bench"), ("glasses", "glass"), ("glass", "glass"),
    ("stories", "story"), ("lamp", "lamp"), ("bus", "bus"),
])
def test_singular(word, expected):
    assert _singular(word) == expected


def test_rank_of_counts_strictly_better_and_shares_ties():
    full = np.array([3.0, 5.0, 1.0, 5.0])
    assert rank_of(full, np.array([0, 1, 2, 3])).tolist() == [3, 1, 4, 1]


def test_all_features_present_and_one_row_per_candidate():
    f = features_for("coffee table")
    assert list(f.columns) == FEATURES
    assert f.index.is_unique


def test_candidates_are_union_of_both_retrievers():
    # depth 1: BM25's best for "sofa" is product 2; dense's best for "sofa" is also 2.
    # For "couch" BM25 finds 3, and dense finds 3 (couch) -> one candidate.
    # For "couch sofa": BM25 top-1 and dense top-1 may differ; union holds both.
    f = features_for("sofa", depth=1)
    assert f.index.tolist() == [2]
    gen = make_gen()
    q = "red couch"
    bm25_top = gen.bm25.search(q, 1)[0][0]
    dense_top = gen.dense.search(q, 1)[0][0]
    got = set(gen.candidates(q, gen.encode_queries([q])[0], depth=1)["product_id"])
    assert got == {bm25_top, dense_top}


def test_name_overlap_handles_plurals_and_phrase_needs_exact_order():
    f = features_for("coffee table")
    assert f.loc[1, "name_overlap"] == 1.0   # "glass coffee tables" -> table matches tables
    assert f.loc[4, "name_overlap"] == 0.5   # "coffee mug"
    assert f.loc[0, "name_phrase"] == 1.0    # "oak coffee table"
    assert f.loc[1, "name_phrase"] == 0.0    # "coffee tables" is not the exact phrase


def test_class_features():
    f = features_for("coffee table")
    assert f.loc[0, "class_overlap"] == 1.0  # "Coffee Tables"
    assert f.loc[4, "class_overlap"] == 0.0  # no class
    assert f.loc[4, "class_vote"] == 0.0     # empty class never gets votes
    assert f.loc[0, "class_vote"] > f.loc[2, "class_vote"]


def test_scores_match_the_retrievers():
    gen = make_gen()
    q = "coffee table"
    f = gen.candidates(q, gen.encode_queries([q])[0]).set_index("product_id")
    bm25 = dict(gen.bm25.search(q, 10))
    assert f.loc[0, "bm25_score"] == pytest.approx(bm25[0])
    assert f.loc[0, "bm25_rank"] == 1
    dense = dict(gen.dense.search(q, 10))
    assert f.loc[0, "cosine"] == pytest.approx(dense[0], abs=1e-6)
    assert np.isnan(f.loc[1, "average_rating"])  # missing rating stays missing
