"""LightGBM LambdaRank re-ranker, trained and scored with 5-fold CV by query.

LambdaRank vs classification: a classifier scores each product on its own
("how relevant is this?") and treats every mistake the same, including
mistakes at rank 150 that nobody sees. LambdaRank looks at pairs of products
for the *same query* and weights each pair by how much NDCG would change if
the two swapped places. Errors near the top of the list cost the most, so
that is where the model puts its effort. Rows are grouped by query so it only
compares products within a query.

No leakage, fold by fold:
- the test fold's queries are never used for training or for choosing the
  number of trees;
- the number of trees is chosen by early stopping on one of the four
  training folds (the inner validation fold), then the model is refit on all
  four training folds with that number;
- other LightGBM settings are fixed up front, not tuned.

Run `python -m src.rerank` after `src.bm25`, `src.dense` and `src.hybrid`.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import N_FOLDS
from src.evaluate import ARTIFACTS_DIR, RUN_COLUMNS
from src.features import FEATURES

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "label_gain": [0, 1, 2],  # same linear gains as src/metrics.py (LightGBM's default is 2^g - 1)
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
MAX_ROUNDS = 2000
EARLY_STOPPING = 100

FEATURE_TABLE = ARTIFACTS_DIR / "features" / "candidates.parquet"
MODEL_DIR = ARTIFACTS_DIR / "models"


FEATURE_GROUPS = {
    "BM25 (score, rank, name score)": ["bm25_score", "bm25_rank", "bm25_name_score"],
    "dense (cosine, rank)": ["cosine", "dense_rank"],
    "text match (name/class overlap, phrase)": ["name_overlap", "name_phrase", "class_overlap"],
    "class vote": ["class_vote"],
    "popularity (rating, reviews)": ["average_rating", "review_count"],
}


def _dataset(frame: pd.DataFrame, features=FEATURES) -> lgb.Dataset:
    # LightGBM needs each query's rows to be contiguous; `group` lists the sizes.
    group = frame.groupby("query_id", sort=False).size().to_numpy()
    return lgb.Dataset(frame[features], label=frame["gain"], group=group, free_raw_data=False)


def fold_masks(frame: pd.DataFrame, folds: dict[int, int], test_fold: int, n_folds: int = N_FOLDS):
    """Row masks (inner_train, inner_valid, test) for one outer fold.

    inner_train + inner_valid are the four training folds; inner_valid is
    only used to pick the number of trees.
    """
    fold = frame["query_id"].map(folds)
    test = fold == test_fold
    inner_valid = fold == (test_fold + 1) % n_folds
    inner_train = ~test & ~inner_valid
    return inner_train, inner_valid, test


def train(frame: pd.DataFrame, n_rounds: int, features=FEATURES) -> lgb.Booster:
    return lgb.train(PARAMS, _dataset(frame, features), n_rounds)


def cross_validate(frame: pd.DataFrame, folds: dict[int, int], n_folds: int = N_FOLDS, features=FEATURES):
    """Out-of-fold model scores for every candidate.

    Returns (frame with a `model_score` column, trees chosen per fold,
    gain importance per fold as a features x folds table).
    """
    frame = frame.sort_values(["query_id", "product_id"]).reset_index(drop=True)
    scores = np.full(len(frame), np.nan)
    best_rounds, importances = {}, {}
    for f in range(n_folds):
        inner_train, inner_valid, test = fold_masks(frame, folds, f, n_folds)
        probe = lgb.train(PARAMS, _dataset(frame[inner_train], features), MAX_ROUNDS,
                          valid_sets=[_dataset(frame[inner_valid], features)],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        best_rounds[f] = max(probe.best_iteration, 1)
        model = train(frame[~test], best_rounds[f], features)
        scores[test.to_numpy()] = model.predict(frame.loc[test, features])
        importances[f] = pd.Series(model.feature_importance("gain"), index=features)
    return frame.assign(model_score=scores), best_rounds, pd.DataFrame(importances)


def ablation(frame: pd.DataFrame, folds: dict[int, int]) -> pd.DataFrame:
    """Cross-validated NDCG@10 with each feature group removed in turn."""
    from src.data import build_qrels, evaluation_query_ids, load_labels
    from src.evaluate import ranked_lists
    from src.metrics import evaluate_run

    qrels = build_qrels(load_labels())
    eval_ids = evaluation_query_ids(qrels)

    def ndcg(features):
        scored, _, _ = cross_validate(frame, folds, features=features)
        return evaluate_run(ranked_lists(to_run(scored)), qrels, eval_ids)["ndcg@10"].mean()

    full = ndcg(FEATURES)
    rows = [{"removed": "(nothing: all features)", "ndcg@10": full, "change": 0.0}]
    for name, group in FEATURE_GROUPS.items():
        score = ndcg([f for f in FEATURES if f not in group])
        rows.append({"removed": name, "ndcg@10": score, "change": score - full})
    return pd.DataFrame(rows).round(4)


def to_run(scored: pd.DataFrame, depth: int = 100) -> pd.DataFrame:
    s = scored.sort_values(["query_id", "model_score", "product_id"], ascending=[True, False, True]).copy()
    s["rank"] = s.groupby("query_id").cumcount() + 1
    return s[s["rank"] <= depth].rename(columns={"model_score": "score"})[RUN_COLUMNS].reset_index(drop=True)


def load_or_build_feature_table(rebuild: bool = False) -> pd.DataFrame:
    from src.data import build_qrels, load_labels, load_products, load_queries
    from src.features import CandidateGenerator, build_feature_table

    if FEATURE_TABLE.exists() and not rebuild:
        frame = pd.read_parquet(FEATURE_TABLE)
        if list(frame.columns) == ["query_id", "product_id", *FEATURES, "gain"]:
            return frame
    gen = CandidateGenerator.build(load_products())
    frame = build_feature_table(gen, load_queries(), build_qrels(load_labels()))
    FEATURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_TABLE, index=False)
    return frame


def main() -> None:
    from src.data import assign_folds, load_queries
    from src.evaluate import evaluate_stage, print_summary, save_run

    frame = load_or_build_feature_table()
    sizes = frame.groupby("query_id").size()
    print(f"{len(frame):,} candidates for {len(sizes)} queries "
          f"(mean {sizes.mean():.0f} per query); gains: "
          f"{frame['gain'].value_counts(normalize=True).sort_index().round(3).to_dict()}")

    folds = assign_folds(load_queries()["query_id"])
    scored, best_rounds, importances = cross_validate(frame, folds)
    print("trees chosen per fold (early stopping on an inner training fold):", best_rounds)

    run = to_run(scored)
    save_run(run, "rerank")
    print_summary("Hybrid + LambdaRank (CV)", evaluate_stage("rerank", run))

    share = (importances / importances.sum()).mul(100)
    table = pd.DataFrame({"gain_%_mean": share.mean(axis=1), "gain_%_fold_std": share.std(axis=1)})
    table = table.sort_values("gain_%_mean", ascending=False).round(1)
    table.to_csv(ARTIFACTS_DIR / "eval" / "feature_importance.csv")
    print("\nfeature importance (share of total split gain, mean over folds):")
    print(table.to_string())

    # Model for the API and demo: all queries, median tree count from CV.
    n_rounds = int(np.median(list(best_rounds.values())))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train(frame.sort_values(["query_id", "product_id"]), n_rounds).save_model(MODEL_DIR / "lambdarank.txt")
    (MODEL_DIR / "lambdarank.json").write_text(json.dumps(
        {"features": FEATURES, "n_rounds": n_rounds, "cv_rounds_by_fold": best_rounds}, indent=2))
    print(f"\nfinal model: {n_rounds} trees on all queries -> {MODEL_DIR / 'lambdarank.txt'}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["ablation"]:
        from src.data import assign_folds, load_queries

        result = ablation(load_or_build_feature_table(), assign_folds(load_queries()["query_id"]))
        result.to_csv(ARTIFACTS_DIR / "eval" / "ablation.csv", index=False)
        print(result.to_string(index=False))
    else:
        main()
