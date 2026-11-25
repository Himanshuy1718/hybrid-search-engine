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
python -m src.dense     # embeds products once (~19 min on a laptop CPU), then dense retrieval
python -m src.hybrid    # reciprocal rank fusion, weight tuned per fold
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

**Dense retrieval** (`src/dense.py`) embeds "name. class. description" with
`all-MiniLM-L6-v2` (384 dimensions, max 256 word pieces, so the name and
class always fit). Vectors are normalised, so FAISS `IndexFlatIP` returns
exact cosine-similarity neighbours; at 43k products exact search takes a
few milliseconds, so an approximate index (HNSW, IVF) would add error for no
gain. Embeddings are cached in `artifacts/embeddings/`.

**Hybrid** (`src/hybrid.py`) uses weighted reciprocal rank fusion,
`w / (60 + rank_bm25) + (1 - w) / (60 + rank_dense)`. RRF uses ranks only, so
BM25 scores and cosine similarities never need to share a scale. The weight
`w` is tuned **per fold on the other four folds** (grid 0.0-1.0), then applied
to the held-out fold; every fold chose 0.4 or 0.5. The API and demo use the
weight picked on all scored queries (0.4), stored in
`artifacts/hybrid_config.json`.

## Results

Mean over the 379 scored queries; in brackets, the standard deviation of the
five per-fold means.

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.704 (0.038) | 0.675 (0.058) | 0.620 (0.033) |
| Dense | 0.703 (0.019) | 0.621 (0.034) | 0.580 (0.033) |
| Hybrid (RRF) | **0.753** (0.018) | **0.686** (0.043) | **0.623** (0.045) |
| Hybrid + LambdaRank | | | |

What the numbers say:

- **Hybrid beats BM25** by +0.050 NDCG@10 (95% bootstrap CI +0.035 to
  +0.065; better on 188 queries, worse on 102; Wilcoxon p = 2e-10).
- **Dense alone ties BM25** on NDCG@10 (difference -0.001, CI -0.022 to
  +0.020) but wins on different queries: of 25,470 Exact products, 6,211 are
  in both top-100s, 3,481 only in BM25's and 2,924 only in dense's. That
  overlap pattern is why fusion helps.
- **Dense makes softer mistakes**: its top 10 have fewer Irrelevant products
  (3% vs 6%) but fewer Exact ones (42% vs 44%). It finds the right kind of
  product more often than the exact one asked for, hence its lower MRR.
- Taking the union of both top-100 lists (up to 200 candidates) would find
  71% of Exact products per query, against 62% for the hybrid's top 100. That
  is the candidate pool for the re-ranker.

Sanity check: a random ranking scores NDCG@10 0.007. BM25's top 10 are 44%
Exact, 36% Partial, 6% Irrelevant and 14% unlabelled. NDCG@10 is fairly high
partly because Partial products earn gain 1, and they make up most labels.
The 14% unlabelled results count as 0; if some are actually relevant, every
stage is under-scored, most of all a stage that finds products the original
labelling never looked at (see Limitations).

## Limitations

- **Unlabelled products count as irrelevant.** WANDS judges a sample of
  products per query, so a retriever that surfaces good products outside that
  sample is penalised. In practice the effect is similar across stages: 14% of
  BM25's top 10 and 15% of dense's top 10 are unlabelled.
