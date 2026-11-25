"""Dense retrieval checks with a tiny fake encoder (no model download)."""
import numpy as np
import pandas as pd

from src.dense import DenseIndex, product_passages

AXES = {"sofa": [1, 0, 0], "couch": [0.9, 0.1, 0], "lamp": [0, 1, 0], "rug": [0, 0, 1]}


class FakeEncoder:
    """Embeds a text as the sum of its known words' vectors."""

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.array([np.sum([AXES.get(w, [0, 0, 0]) for w in t.split()], axis=0) for t in texts],
                       dtype=np.float32)
        if normalize_embeddings:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


def make_index():
    enc = FakeEncoder()
    texts = ["sofa", "lamp", "rug", "sofa lamp"]
    return DenseIndex(enc.encode(texts), product_ids=[100, 101, 102, 103], encoder=enc)


def test_nearest_product_by_meaning_not_words():
    # "couch" shares no word with "sofa" but its vector is close.
    hits = make_index().search("couch", k=4)
    assert hits[0][0] == 100
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_scores_are_cosine_similarities():
    hits = dict(make_index().search("sofa", k=4))
    assert hits[100] == np.float32(1.0)
    assert np.isclose(hits[103], 1 / np.sqrt(2), atol=1e-6)  # halfway between sofa and lamp
    assert np.isclose(hits[101], 0.0, atol=1e-6)


def test_k_larger_than_index_returns_only_real_products():
    hits = make_index().search("rug", k=10)
    assert sorted(pid for pid, _ in hits) == [100, 101, 102, 103]


def test_passages_put_name_first_and_skip_empty_fields():
    products = pd.DataFrame({"product_name": ["oak desk", "lamp"],
                             "product_class": ["Desks", ""],
                             "product_description": ["solid oak .", ""]})
    assert product_passages(products) == ["oak desk. Desks. solid oak .", "lamp"]
