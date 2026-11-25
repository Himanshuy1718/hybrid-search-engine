import pandas as pd
import pytest

from src.evaluate import ranked_lists, summarize_by_fold


def test_ranked_lists_orders_by_rank_not_row_order():
    run = pd.DataFrame(
        {"query_id": [1, 1, 2, 1], "product_id": [7, 5, 9, 6], "rank": [3, 1, 1, 2], "score": [0.1, 0.9, 1.0, 0.5]}
    )
    assert ranked_lists(run) == {1: [5, 6, 7], 2: [9]}


def test_summarize_by_fold_mean_and_spread():
    per_query = pd.DataFrame({"ndcg@10": [1.0, 0.0, 0.5, 0.5]}, index=[1, 2, 3, 4])
    folds = {1: 0, 2: 0, 3: 1, 4: 1}  # fold means: 0.5 and 0.5
    out = summarize_by_fold(per_query, folds)
    assert out.loc["ndcg@10", "mean"] == pytest.approx(0.5)
    assert out.loc["ndcg@10", "fold_std"] == pytest.approx(0.0)
