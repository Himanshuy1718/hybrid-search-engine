"""Checks for the data helpers that do not need the real dataset."""
import pandas as pd
import pytest

from src.data import assign_folds, build_qrels, evaluation_query_ids, load_labels


def test_build_qrels_maps_query_to_product_gains():
    labels = pd.DataFrame(
        {"query_id": [0, 0, 1], "product_id": [10, 11, 10], "gain": [2, 0, 1]}
    )
    assert build_qrels(labels) == {0: {10: 2, 11: 0}, 1: {10: 1}}


def test_evaluation_queries_need_an_exact_product():
    qrels = {0: {10: 2}, 1: {10: 1, 11: 0}, 2: {12: 0, 13: 2}}
    assert evaluation_query_ids(qrels) == [0, 2]


def test_folds_cover_every_query_once_and_are_balanced():
    folds = assign_folds(range(103), n_folds=5)
    assert sorted(folds) == list(range(103))
    sizes = pd.Series(folds).value_counts()
    assert sizes.max() - sizes.min() <= 1


def test_folds_are_deterministic_and_order_independent():
    assert assign_folds([5, 1, 9, 3]) == assign_folds([9, 3, 1, 5])


def test_load_labels_rejects_unknown_labels(tmp_path):
    (tmp_path / "label.csv").write_text(
        "id\tquery_id\tproduct_id\tlabel\n0\t0\t1\tMaybe\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected labels"):
        load_labels(tmp_path)


def test_duplicate_labels_resolve_by_majority_then_lower_gain(tmp_path):
    rows = [
        (0, 1, "Exact"), (0, 1, "Exact"),                       # agree -> Exact
        (0, 2, "Exact"), (0, 2, "Partial"),                     # tie -> lower: Partial
        (0, 3, "Exact"), (0, 3, "Partial"), (0, 3, "Partial"),  # majority -> Partial
        (0, 4, "Partial"), (0, 4, "Exact"), (0, 4, "Exact"),    # majority -> Exact
        (1, 1, "Irrelevant"),                                   # single label kept
    ]
    lines = ["id\tquery_id\tproduct_id\tlabel"]
    lines += [f"{i}\t{q}\t{p}\t{lab}" for i, (q, p, lab) in enumerate(rows)]
    (tmp_path / "label.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = load_labels(tmp_path)
    assert not labels.duplicated(["query_id", "product_id"]).any()
    got = {(q, p): lab for q, p, lab in labels[["query_id", "product_id", "label"]].itertuples(index=False)}
    assert got == {(0, 1): "Exact", (0, 2): "Partial", (0, 3): "Partial",
                   (0, 4): "Exact", (1, 1): "Irrelevant"}
