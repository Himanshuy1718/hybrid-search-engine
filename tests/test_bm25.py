"""BM25 checks on a tiny corpus worked out by hand (k1 = 1.2, b = 0.75)."""
import math

import pandas as pd
import pytest

from src.bm25 import BM25Index, tokenize


def make_products(names, descriptions=None):
    n = len(names)
    return pd.DataFrame(
        {
            "product_id": [10 + i for i in range(n)],
            "product_name": names,
            "product_class": [""] * n,
            "product_description": descriptions or [""] * n,
        }
    )


# A=10 "red sofa", B=11 "blue sofa bed", C=12 "red chair", D=13 "oak table"
# N = 4 docs, lengths 2, 3, 2, 2 -> avgdl = 9 / 4 = 2.25
PRODUCTS = make_products(["red sofa", "blue sofa bed", "red chair", "oak table"])


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("All-Clad 7 qt. Slow-Cooker!") == ["all", "clad", "7", "qt", "slow", "cooker"]


def test_scores_by_hand():
    index = BM25Index(PRODUCTS)
    s = dict(index.search("red sofa", k=10))
    # "red" and "sofa" each appear in 2 of 4 docs: IDF = ln(1 + 2.5 / 2.5) = ln 2
    # A (len 2): per term 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.25)) = 2.2 / 2.1
    assert s[10] == pytest.approx(2 * math.log(2) * 2.2 / 2.1)  # 1.4523
    # C (len 2) matches only "red"
    assert s[12] == pytest.approx(math.log(2) * 2.2 / 2.1)  # 0.7262
    # B (len 3) matches only "sofa": 2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 2.25)) = 2.2 / 2.5
    assert s[11] == pytest.approx(math.log(2) * 2.2 / 2.5)  # 0.6100


def test_ranking_order_and_non_matches_excluded():
    index = BM25Index(PRODUCTS)
    # A matches both words; C beats B because a shorter product with one
    # match ranks above a longer one (length normalisation). D matches nothing.
    assert [pid for pid, _ in index.search("red sofa", k=10)] == [10, 12, 11]


def test_rarer_terms_get_higher_idf_and_idf_stays_positive():
    products = make_products(["sofa", "sofa chair", "sofa bed", "sofa oak"])
    index = BM25Index(products)
    idf = {t: index.idf[index.vocab[t]] for t in ["oak", "sofa"]}
    assert idf["oak"] > idf["sofa"] > 0  # "sofa" is in every product, still > 0


def test_term_frequency_saturates():
    products = make_products(["lamp", "lamp lamp", "lamp lamp lamp lamp", "shade"])
    index = BM25Index(products, b=0.0)  # switch off length effects to isolate tf
    s = dict(index.search("lamp", k=10))
    gain_1_to_2 = s[11] - s[10]
    gain_2_to_4 = s[12] - s[11]
    assert gain_1_to_2 > 0 and gain_2_to_4 > 0
    assert gain_2_to_4 < gain_1_to_2  # doubling tf again adds less


def test_description_words_are_searchable():
    products = make_products(["chair", "table"], ["solid walnut frame", "glass top"])
    index = BM25Index(products)
    assert [pid for pid, _ in index.search("walnut", k=10)] == [10]


def test_unknown_or_empty_query_returns_nothing():
    index = BM25Index(PRODUCTS)
    assert index.search("zebra", k=10) == []
    assert index.search("   ", k=10) == []


def test_k_limits_results_and_ties_break_by_product_id():
    products = make_products(["desk", "desk", "desk", "chair"])
    index = BM25Index(products)
    assert [pid for pid, _ in index.search("desk", k=2)] == [10, 11]
