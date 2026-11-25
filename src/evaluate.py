"""Shared evaluation harness. Every stage is scored through `evaluate_stage`,
so all stages use the same queries, judgments, metric code and folds.

A run is a DataFrame with one row per retrieved product:
    query_id, product_id, rank (1 = best), score
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.data import assign_folds, build_qrels, evaluation_query_ids, load_labels, load_queries
from src.metrics import evaluate_run

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUN_COLUMNS = ["query_id", "product_id", "rank", "score"]


def save_run(run: pd.DataFrame, name: str) -> Path:
    path = ARTIFACTS_DIR / "runs" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    run[RUN_COLUMNS].to_parquet(path, index=False)
    return path


def load_run(name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "runs" / f"{name}.parquet")


def ranked_lists(run: pd.DataFrame) -> dict[int, list[int]]:
    run = run.sort_values(["query_id", "rank"])
    return {int(q): g["product_id"].astype(int).tolist() for q, g in run.groupby("query_id", sort=False)}


def summarize_by_fold(per_query: pd.DataFrame, folds: Mapping[int, int]) -> pd.DataFrame:
    """Mean of each metric over all scored queries, plus the standard
    deviation of the five per-fold means (how much the score moves with
    the particular set of test queries)."""
    fold_of = per_query.index.map(folds)
    fold_means = per_query.groupby(fold_of).mean()
    return pd.DataFrame({"mean": per_query.mean(), "fold_std": fold_means.std(ddof=1)})


def evaluate_stage(name: str, run: pd.DataFrame) -> pd.DataFrame:
    """Score a run, save per-query metrics and add the stage to results.csv."""
    queries, labels = load_queries(), load_labels()
    qrels = build_qrels(labels)
    folds = assign_folds(queries["query_id"])
    per_query = evaluate_run(ranked_lists(run), qrels, evaluation_query_ids(qrels))

    eval_dir = ARTIFACTS_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(eval_dir / f"{name}_per_query.csv")

    summary = summarize_by_fold(per_query, folds)
    _record(name, summary)
    return summary


def _record(name: str, summary: pd.DataFrame) -> None:
    path = ARTIFACTS_DIR / "results.csv"
    row = {"stage": name}
    for metric, values in summary.iterrows():
        row[metric] = round(values["mean"], 4)
        row[f"{metric}_fold_std"] = round(values["fold_std"], 4)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["stage"] != name], new], ignore_index=True)
    new.to_csv(path, index=False)


def print_summary(label: str, summary: pd.DataFrame) -> None:
    cells = [f"{m} {r['mean']:.4f} (fold std {r['fold_std']:.4f})" for m, r in summary.iterrows()]
    print(f"{label}: " + " | ".join(cells))
