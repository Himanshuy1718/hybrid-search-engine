# Hybrid Semantic Search and Ranking Engine

A two-stage product search system on the [WANDS](https://github.com/wayfair/WANDS)
dataset (Wayfair): keyword (BM25) and embedding (FAISS) retrieval, merged and
re-ranked with LightGBM LambdaRank. Each stage is measured against plain
keyword search.

```
query -> BM25 + embedding search (FAISS) -> merged candidates -> LightGBM LambdaRank -> top 10
```

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `product.csv`, `query.csv` and `label.csv` from
`https://github.com/wayfair/WANDS/tree/main/dataset` into `data/`.
They are tab-separated despite the `.csv` extension.

```powershell
python -m src.data      # data summary
pytest                  # tests
python -m src.bm25      # BM25 baseline -> artifacts/runs/bm25.parquet, artifacts/results.csv
```

Each stage writes its top-100 run to `artifacts/runs/`, per-query metrics to
`artifacts/eval/`, and its summary row to `artifacts/results.csv`.

## Data notes

- 42,994 products, 480 queries, 233,448 raw labels (Exact 11%, Partial 63%, Irrelevant 26%).
- 1,467 (query, product) pairs are labelled more than once; only 14 disagree.
  Resolved by majority vote, ties to the lower gain -> 231,873 labels.
- 101 queries have no Exact product, so 379 queries are scored. All 480 are
  assigned to folds and the unscored ones are still used for training.
- Some queries have hundreds of Exact products (max 876), so Recall@100 cannot
  reach 1 for them.
- 14% of products have no description, 6.6% no product class; 22% have no
  ratings (`average_rating` is left missing rather than set to 0).
- 72 descriptions contain mis-encoded characters from the source file; left as is.

## Evaluation rules

These are fixed before any model is built and apply to every stage.

| Rule | Choice |
|---|---|
| Relevance gains | Exact = 2, Partial = 1, Irrelevant = 0 |
| Unlabelled products | Count as gain 0. WANDS labels only a sample of products per query, so some "misses" may really be relevant. Absolute scores are therefore conservative. The comparison between stages is still fair because all stages are scored the same way. |
| "Relevant" for MRR and Recall | Exact only. Partial does not count. |
| NDCG | NDCG@10, linear gain, log2 discount. The ideal ranking uses all labelled products for the query, not only the retrieved ones. |
| MRR | MRR@10: 1 / position of the first Exact product in the top 10, else 0. |
| Recall | Recall@100: share of the query's Exact products found in the top 100. |
| Query set | Queries with at least one Exact product (MRR and Recall are undefined otherwise). The same queries and the same metric code (`src/metrics.py`) score every stage. |
| Duplicate labels | Majority vote per (query, product); ties go to the lower gain. |
| Splits | 5-fold cross-validation **by query**, never by query-product pair, so no test query is seen in training. Folds are fixed with a seed (`src/data.py`) and cover all 480 queries. |
| Tuning | Fusion weights and model parameters are tuned on training folds only. |

## Design choices

**BM25 is implemented directly** (`src/bm25.py`) instead of with the
`rank-bm25` package. That package's IDF goes negative for words in more than
half the documents and is then floored to a constant, so a filler word in 90%
of products ("the", "with") gets *more* weight than a real word in 50%. The
implementation here uses the Lucene/Elasticsearch IDF,
`ln(1 + (N - n + 0.5) / (n + 0.5))`, which is always positive and always
favours rarer words. It precomputes a sparse product-by-term weight matrix,
so a query is a sum of a few columns (about 1.5 ms per query on CPU).
Settings: k1 = 1.2, b = 0.75 (Lucene defaults, not tuned); fields are product
name, class and description joined; lowercase alphanumeric tokens, no
stemming or stopword removal.

## Results

Mean over the 379 scored queries; in brackets, the standard deviation of the
five per-fold means.

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.704 (0.038) | 0.675 (0.058) | 0.620 (0.033) |
| Dense | | | |
| Hybrid | | | |
| Hybrid + LambdaRank | | | |

The re-ranker only reorders the hybrid candidates, so its Recall@100 equals the
hybrid stage's by design.

Sanity check: a random ranking scores NDCG@10 0.007. BM25's top 10 are 44%
Exact, 36% Partial, 6% Irrelevant and 14% unlabelled. NDCG@10 is fairly high
partly because Partial products earn gain 1, and they make up most labels.
The 14% unlabelled results count as 0; if some are actually relevant, every
stage is under-scored, most of all a stage that finds products the original
labelling never looked at (see Limitations).

## Limitations

- **Unlabelled products count as irrelevant.** WANDS judges a sample of
  products per query. A retriever that surfaces good products outside that
  sample is penalised. This may affect embedding search more than BM25.
