"""BM25 keyword retrieval over product name, class and description.

How BM25 scores a product d for a query q:

    score(q, d) = sum over query terms t of
                  IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

- tf(t, d): how often t appears in d. The k1 term makes this saturate: the
  5th "sofa" adds much less than the 1st, so keyword stuffing doesn't win.
- |d| / avgdl: document length relative to the average. b controls how much
  long documents are penalised (a match in a 10-word title says more than
  the same match in a 300-word description).
- IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)), with N documents and n_t of
  them containing t. Rare words count more than common ones.

Why not the rank-bm25 package: its IDF goes negative for words in more than
half the documents and is then floored to a constant, which gives a filler
word in 90% of products ("the") *more* weight than a real word in 50%. The
Lucene/Elasticsearch IDF above is always positive and always favours rarer
words. k1 = 1.2 and b = 0.75 are the Lucene defaults and are not tuned.

Run `python -m src.bm25` to retrieve the top 100 for every query and score it.
"""
from __future__ import annotations

import re
from collections import Counter
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

INDEX_FIELDS = ("product_name", "product_class", "product_description")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and keep runs of letters/digits. No stemming or stopwords."""
    return _TOKEN_RE.findall(text.lower())


def product_text(products: pd.DataFrame, fields=INDEX_FIELDS) -> pd.Series:
    return products[list(fields)].fillna("").astype(str).agg(" ".join, axis=1)


class BM25Index:
    def __init__(self, products: pd.DataFrame, fields=INDEX_FIELDS, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.product_ids = products["product_id"].to_numpy()
        self.row_of = {int(pid): i for i, pid in enumerate(self.product_ids)}

        vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = vectorizer.fit_transform(product_text(products, fields)).tocsr().astype(np.float64)
        self.vocab: dict[str, int] = vectorizer.vocabulary_

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        doc_freq = np.bincount(tf.indices, minlength=tf.shape[1])  # docs containing each term
        self.idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Precompute the full BM25 weight of every (product, term) pair once.
        # A query is then just "add up the columns for its terms".
        length_norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        weights = tf.data * (k1 + 1) / (tf.data + length_norm[rows]) * self.idf[tf.indices]
        self.weights = sp.csr_matrix((weights, tf.indices, tf.indptr), shape=tf.shape).tocsc()

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every product (0 where no query term matches)."""
        counts = Counter(t for t in tokenize(query) if t in self.vocab)
        if not counts:
            return np.zeros(len(self.product_ids))
        cols = [self.vocab[t] for t in counts]
        query_tf = np.fromiter(counts.values(), dtype=np.float64)
        return np.asarray(self.weights[:, cols] @ query_tf).ravel()

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        """Top k (product_id, score), best first. Products with no matching term are not returned.

        Ties are broken by product_id so results are deterministic.
        """
        s = self.scores(query)
        hits = np.flatnonzero(s > 0)
        order = hits[np.lexsort((self.product_ids[hits], -s[hits]))][:k]
        return [(int(self.product_ids[i]), float(s[i])) for i in order]


def main(k: int = 100) -> None:
    from src.data import load_products, load_queries
    from src.evaluate import RUN_COLUMNS, evaluate_stage, print_summary, save_run

    products, queries = load_products(), load_queries()

    t0 = perf_counter()
    index = BM25Index(products)
    print(f"indexed {len(products):,} products, {len(index.vocab):,} terms "
          f"in {perf_counter() - t0:.1f}s")

    rows = []
    t0 = perf_counter()
    for qid, text in zip(queries["query_id"], queries["query"]):
        for rank, (pid, score) in enumerate(index.search(text, k), start=1):
            rows.append((int(qid), pid, rank, score))
    per_query_ms = (perf_counter() - t0) / len(queries) * 1000
    run = pd.DataFrame(rows, columns=RUN_COLUMNS)
    short = (run.groupby("query_id").size() < k).sum() + (len(queries) - run.query_id.nunique())
    print(f"searched {len(queries)} queries, {per_query_ms:.1f} ms/query on average; "
          f"{short} returned fewer than {k} results")

    save_run(run, "bm25")
    print_summary("BM25", evaluate_stage("bm25", run))


if __name__ == "__main__":
    main()
