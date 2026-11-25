import lightgbm as lgb
import numpy as np
import pytest

import pandas as pd

from src.engine import SearchEngine
from src.features import FEATURES
from src.hybrid import rrf_fuse
from tests.fakes import PRODUCTS, make_gen


def _as_run(hits):
    return pd.DataFrame({"query_id": 0, "product_id": [p for p, _ in hits],
                         "rank": range(1, len(hits) + 1), "score": [s for _, s in hits]})


@pytest.fixture(scope="module")
def engine():
    rng = np.random.default_rng(0)
    x = rng.random((200, len(FEATURES)))
    y = rng.integers(0, 3, 200)
    booster = lgb.train({"objective": "lambdarank", "label_gain": [0, 1, 2], "min_data_in_leaf": 5,
                         "verbose": -1, "seed": 0}, lgb.Dataset(x, y, group=[20] * 10), 10)
    return SearchEngine(PRODUCTS, make_gen(), booster, w_bm25=0.4, rrf_k=60)


def test_bm25_mode_matches_bm25(engine):
    hits, timings = engine.search("coffee table", k=3, mode="bm25")
    assert hits == engine.gen.bm25.search("coffee table", 3)
    assert timings["total"] >= 0


def test_hybrid_mode_matches_offline_fusion(engine):
    q = "coffee table"
    hits, _ = engine.search(q, k=4, mode="hybrid")
    bm25 = _as_run(engine.gen.bm25.search(q, 100))
    dense = _as_run(engine.gen.dense.search(q, 100))
    expected = rrf_fuse(bm25, dense, 0.4, 60, depth=4)
    assert [p for p, _ in hits] == expected["product_id"].tolist()


def test_rerank_mode_orders_by_model_score(engine):
    hits, timings = engine.search("coffee table", k=3, mode="rerank")
    assert len(hits) == 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert {"encode_query", "retrieve_and_features", "rerank_model", "total"} <= set(timings)


def test_describe_fills_details_and_missing_rating(engine):
    rows = engine.describe([(1, 0.5), (0, 0.25)])
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["name"] == "glass coffee tables"
    assert rows[0]["average_rating"] is None  # NaN -> None for JSON
    assert rows[1]["review_count"] == 10


def test_unknown_mode_rejected(engine):
    with pytest.raises(ValueError):
        engine.search("sofa", mode="magic")
