"""Load and clean the WANDS dataset, build relevance judgments and query folds.

Run `python -m src.data` for a quick summary of the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
TEXT_COLUMNS = [
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
]
N_FOLDS = 5
FOLD_SEED = 42


def _read(name: str, data_dir: Path) -> pd.DataFrame:
    # WANDS files are tab-separated despite the .csv extension.
    return pd.read_csv(data_dir / name, sep="\t", encoding="utf-8")


def load_products(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read("product.csv", data_dir)
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.rename(columns={"category hierarchy": "category_hierarchy"})
    # No ratings is recorded as missing; a count of 0 is the honest value.
    for col in ["rating_count", "review_count"]:
        df[col] = df[col].fillna(0).astype(int)
    # average_rating stays NaN when there are no ratings: LightGBM handles
    # missing values natively, and 0 stars would be a false signal.
    return df.set_index("product_id", drop=False)


def load_queries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read("query.csv", data_dir)
    df["query"] = df["query"].fillna("").astype(str).str.strip()
    df["query_class"] = df["query_class"].fillna("").astype(str).str.strip()
    return df.set_index("query_id", drop=False)


def resolve_duplicate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one label per (query, product) pair.

    WANDS has ~1.5k pairs labelled more than once; almost all copies agree.
    For the few that disagree, take the most common gain, and on a tie the
    *lower* gain, so a ranker is never credited for a product that one rater
    judged less relevant.
    """
    counts = df.groupby(["query_id", "product_id", "gain"]).size().rename("n").reset_index()
    counts = counts.sort_values(["query_id", "product_id", "n", "gain"],
                                ascending=[True, True, False, True])
    resolved = counts.drop_duplicates(["query_id", "product_id"], keep="first")
    resolved["label"] = resolved["gain"].map({v: k for k, v in GAIN.items()})
    return resolved[["query_id", "product_id", "label", "gain"]].reset_index(drop=True)


def load_labels(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read("label.csv", data_dir)
    unknown = set(df["label"]) - set(GAIN)
    if unknown:
        raise ValueError(f"unexpected labels: {unknown}")
    df["gain"] = df["label"].map(GAIN).astype(int)
    return resolve_duplicate_labels(df)


def build_qrels(labels: pd.DataFrame) -> dict[int, dict[int, int]]:
    """query_id -> {product_id: gain}. Unlabelled products are simply absent (gain 0)."""
    qrels: dict[int, dict[int, int]] = {}
    for qid, pid, gain in labels[["query_id", "product_id", "gain"]].itertuples(index=False):
        qrels.setdefault(int(qid), {})[int(pid)] = int(gain)
    return qrels


def evaluation_query_ids(qrels: dict[int, dict[int, int]]) -> list[int]:
    """Queries with at least one Exact product.

    MRR and Recall are undefined without an Exact product, so these are the
    queries every stage is scored on.
    """
    return sorted(q for q, judged in qrels.items() if any(g == 2 for g in judged.values()))


def assign_folds(query_ids, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[int, int]:
    """Assign each query (not each query-product pair) to one of n_folds folds.

    Splitting by query means no query appears in both training and test data,
    so the re-ranker cannot memorise answers for the queries it is tested on.
    """
    ids = np.array(sorted(query_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {int(q): i % n_folds for i, q in enumerate(ids)}


def summarize(data_dir: Path = DATA_DIR) -> None:
    products = load_products(data_dir)
    queries = load_queries(data_dir)
    raw_labels = _read("label.csv", data_dir)
    labels = load_labels(data_dir)
    qrels = build_qrels(labels)
    eval_ids = evaluation_query_ids(qrels)

    print(f"products: {len(products):,}   queries: {len(queries):,}   "
          f"labels: {len(raw_labels):,} raw -> {len(labels):,} after de-duplication")
    missing_p = set(labels.product_id) - set(products.product_id)
    missing_q = set(labels.query_id) - set(queries.query_id)
    print(f"labels pointing at unknown products: {len(missing_p)}, unknown queries: {len(missing_q)}")
    print("\nempty text fields (share of products):")
    for col in ["product_name", "product_class", "category_hierarchy",
                "product_description", "product_features"]:
        print(f"  {col:<22} {(products[col] == '').mean():6.1%}")
    print(f"  {'average_rating NaN':<22} {products['average_rating'].isna().mean():6.1%}")

    print("\nlabel balance:")
    print(labels["label"].value_counts(normalize=True).round(3).to_string())

    per_query = labels.groupby("query_id")["gain"]
    n_exact = labels[labels.gain == 2].groupby("query_id").size().reindex(queries.index, fill_value=0)
    print("\nlabels per query:", per_query.size().describe()[["min", "50%", "max"]].to_dict())
    print("Exact per query:", n_exact.describe()[["min", "50%", "max"]].to_dict())
    print(f"queries with no Exact product: {(n_exact == 0).sum()}")
    print(f"queries used for evaluation: {len(eval_ids)}")
    print(f"labelled products: {labels.product_id.nunique():,} of {len(products):,}")

    # Folds cover every query: queries without an Exact product are still
    # useful training data for the re-ranker (Partial vs Irrelevant), they
    # are just never scored.
    folds = assign_folds(queries.query_id)
    print("queries per fold (all):", np.bincount(list(folds.values())).tolist())
    print("scored queries per fold:", np.bincount([folds[q] for q in eval_ids]).tolist())


if __name__ == "__main__":
    summarize()
