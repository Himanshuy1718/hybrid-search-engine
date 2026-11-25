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
```

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

## Results

_Filled in from the runs as each phase is completed._

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | | | |
| Dense | | | |
| Hybrid | | | |
| Hybrid + LambdaRank | | | |

The re-ranker only reorders the hybrid candidates, so its Recall@100 equals the
hybrid stage's by design.
