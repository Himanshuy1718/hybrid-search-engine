"""Re-ranker checks on synthetic data: it learns, and it never peeks at the test fold."""
import numpy as np
import pandas as pd

from src.features import FEATURES
from src.metrics import ndcg_at_k
from src.rerank import cross_validate, fold_masks, to_run

N_QUERIES, PER_QUERY = 60, 20


def synthetic_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(N_QUERIES):
        x = rng.random((PER_QUERY, len(FEATURES)))
        signal = x[:, FEATURES.index("name_overlap")]
        gain = np.digitize(signal, [0.6, 0.85])  # 0, 1 or 2, driven by one feature
        for pid in range(PER_QUERY):
            rows.append({"query_id": q, "product_id": pid,
                         **dict(zip(FEATURES, x[pid])), "gain": int(gain[pid])})
    return pd.DataFrame(rows)


FOLDS = {q: q % 5 for q in range(N_QUERIES)}


def test_fold_masks_split_by_query_without_overlap():
    frame = synthetic_frame()
    for f in range(5):
        inner_train, inner_valid, test = fold_masks(frame, FOLDS, f)
        sets = [set(frame.loc[m, "query_id"]) for m in (inner_train, inner_valid, test)]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
        assert sets[2] == {q for q, fold in FOLDS.items() if fold == f}
        assert (inner_train | inner_valid | test).all()


def test_model_learns_the_signal():
    frame = synthetic_frame()
    scored, _, importances = cross_validate(frame, FOLDS)
    run = to_run(scored)
    ndcgs = []
    for q, g in run.groupby("query_id"):
        judged = dict(zip(frame.loc[frame.query_id == q, "product_id"], frame.loc[frame.query_id == q, "gain"]))
        if any(judged.values()):
            ndcgs.append(ndcg_at_k(g.sort_values("rank")["product_id"].tolist(), judged, 10))
    assert np.mean(ndcgs) > 0.9
    assert importances.mean(axis=1).idxmax() == "name_overlap"


def test_test_fold_scores_do_not_depend_on_test_fold_labels():
    # Scramble the labels of fold 0's queries. If fold 0's predictions change,
    # the model for fold 0 must have trained on (or tuned with) fold 0.
    frame = synthetic_frame()
    scrambled = frame.copy()
    in_fold0 = scrambled["query_id"].map(FOLDS) == 0
    scrambled.loc[in_fold0, "gain"] = np.random.default_rng(1).permutation(scrambled.loc[in_fold0, "gain"].to_numpy())

    a, _, _ = cross_validate(frame, FOLDS)
    b, _, _ = cross_validate(scrambled, FOLDS)
    key = ["query_id", "product_id"]
    a0 = a[a["query_id"].map(FOLDS) == 0].sort_values(key)["model_score"].to_numpy()
    b0 = b[b["query_id"].map(FOLDS) == 0].sort_values(key)["model_score"].to_numpy()
    assert np.array_equal(a0, b0)
    # ...while the other folds, which did train on fold 0, do change.
    a1 = a[a["query_id"].map(FOLDS) == 1].sort_values(key)["model_score"].to_numpy()
    b1 = b[b["query_id"].map(FOLDS) == 1].sort_values(key)["model_score"].to_numpy()
    assert not np.array_equal(a1, b1)
