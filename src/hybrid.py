"""Hybrid retrieval: merge BM25 and dense results with reciprocal rank fusion.

Weighted RRF score of a product:

    score = w / (k + rank_bm25) + (1 - w) / (k + rank_dense)

A list the product does not appear in contributes nothing. RRF uses only
ranks, not raw scores, so there is no need to put BM25 scores (0 to ~30) and
cosine similarities (-1 to 1) on the same scale. k = 60 is the usual value
from the original RRF paper; it flattens the gap between rank 1 and rank 2 so
one list can't dominate on its top hit alone.

The weight w is tuned without leakage: for each test fold, w is the value
that maximises NDCG@10 on the *other four* folds, and only then is it
applied to the test fold. The weight for the deployed system (API, demo) is
picked on all scored queries, after the cross-validated score is recorded.

Run `python -m src.hybrid` after `src.bm25` and `src.dense`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluate import ARTIFACTS_DIR, RUN_COLUMNS, ranked_lists
from src.metrics import evaluate_run

RRF_K = 60
WEIGHTS = [round(w, 1) for w in np.linspace(0, 1, 11)]  # weight on BM25
CONFIG_PATH = ARTIFACTS_DIR / "hybrid_config.json"


def rrf_fuse(bm25: pd.DataFrame, dense: pd.DataFrame, w_bm25: float,
             k: int = RRF_K, depth: int = 100) -> pd.DataFrame:
    """Fuse two runs into one run of the top `depth` products per query."""
    parts = []
    for run, weight in ((bm25, w_bm25), (dense, 1.0 - w_bm25)):
        if weight > 0:
            parts.append(run[["query_id", "product_id"]].assign(score=weight / (k + run["rank"])))
    fused = (pd.concat(parts).groupby(["query_id", "product_id"], as_index=False)["score"].sum()
             .sort_values(["query_id", "score", "product_id"], ascending=[True, False, True]))
    fused["rank"] = fused.groupby("query_id").cumcount() + 1
    return fused[fused["rank"] <= depth][RUN_COLUMNS].reset_index(drop=True)


def pick_weight(ndcg_by_weight: pd.DataFrame, train_ids) -> float:
    """Weight with the best mean NDCG@10 on the training queries only.

    ndcg_by_weight: one row per scored query, one column per candidate weight.
    Ties go to the first (lowest) weight.
    """
    return float(ndcg_by_weight.loc[list(train_ids)].mean().idxmax())


def cross_validated_hybrid(bm25, dense, qrels, eval_ids, folds, weights=WEIGHTS):
    """Fused run where each query uses the weight chosen without its own fold."""
    fused = {w: rrf_fuse(bm25, dense, w) for w in weights}
    ndcg_by_weight = pd.DataFrame(
        {w: evaluate_run(ranked_lists(run), qrels, eval_ids)["ndcg@10"] for w, run in fused.items()}
    )
    chosen, pieces = {}, []
    for fold in sorted(set(folds.values())):
        train_ids = [q for q in eval_ids if folds[q] != fold]
        w = chosen[fold] = pick_weight(ndcg_by_weight, train_ids)
        test_queries = {q for q, f in folds.items() if f == fold}
        pieces.append(fused[w][fused[w]["query_id"].isin(test_queries)])
    return pd.concat(pieces, ignore_index=True), chosen, ndcg_by_weight


def main() -> None:
    from src.data import assign_folds, build_qrels, evaluation_query_ids, load_labels, load_queries
    from src.evaluate import evaluate_stage, load_run, print_summary, save_run

    qrels = build_qrels(load_labels())
    eval_ids = evaluation_query_ids(qrels)
    folds = assign_folds(load_queries()["query_id"])
    bm25, dense = load_run("bm25"), load_run("dense")

    run, chosen, ndcg_by_weight = cross_validated_hybrid(bm25, dense, qrels, eval_ids, folds)
    print("BM25 weight chosen per fold (on the other 4 folds):", chosen)
    print("mean NDCG@10 by BM25 weight, all scored queries (for reference only):")
    print(ndcg_by_weight.mean().round(4).to_string())

    save_run(run, "hybrid")
    print_summary("Hybrid (CV)", evaluate_stage("hybrid", run))

    deploy_w = pick_weight(ndcg_by_weight, eval_ids)
    CONFIG_PATH.write_text(json.dumps({"w_bm25": deploy_w, "rrf_k": RRF_K,
                                       "cv_weights_by_fold": chosen}, indent=2))
    print(f"deployment weight (all scored queries): {deploy_w} -> {CONFIG_PATH.name}")


if __name__ == "__main__":
    main()
