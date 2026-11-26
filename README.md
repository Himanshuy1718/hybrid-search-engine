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

```powershell
python -m src.download  # WANDS CSVs from a pinned commit into data/, SHA-256 checked
python -m src.data      # data summary
pytest                  # tests
python -m src.bm25      # BM25 baseline -> artifacts/runs/bm25.parquet, artifacts/results.csv
python -m src.dense     # embeds products once (~19 min on a laptop CPU), then dense retrieval
python -m src.hybrid    # reciprocal rank fusion, weight tuned per fold
python -m src.rerank    # candidate features + LightGBM LambdaRank, 5-fold CV, final model
python -m src.rerank ablation   # CV score with each feature group removed
```

Each stage writes its top-100 run to `artifacts/runs/`, per-query metrics to
`artifacts/eval/`, and its summary row to `artifacts/results.csv`.

## API and demo

After the pipeline above has produced `artifacts/` (embeddings, fusion
weight, model):

```powershell
uvicorn src.api:app --port 8000     # API; http://localhost:8000 opens the interactive docs
streamlit run app.py                # demo page on http://localhost:8501
python -m src.latency               # latency benchmark against the running API
```

`GET /search?q=modern farmhouse dining table&k=5` (mode defaults to `rerank`;
also `hybrid`, `dense`, `bm25`):

```json
{
  "query": "modern farmhouse dining table", "mode": "rerank", "k": 5, "took_ms": 32.45,
  "timings_ms": {"encode_query": 18.24, "retrieve_and_features": 12.07, "rerank_model": 2.14, "total": 32.45},
  "results": [
    {"rank": 1, "product_id": 14115, "name": "modern farmhouse solid wood dining table",
     "product_class": "Dining Tables", "score": 4.132784, "average_rating": null, "review_count": 0},
    ...
  ]
}
```

Everything is loaded once at startup (about 30 s); a request does no disk
I/O. The online code path reuses the offline feature code
(`CandidateGenerator`), so the model sees identical features in training and
serving. Online fusion uses a plain-Python RRF (`rrf_fuse_hits`) that a test
checks is identical to the offline pandas version.

The demo searches with any stage, can show BM25 side by side, and, when the
query is one of the 480 WANDS queries, colours each result by its human label.

### Latency

Each of the 480 WANDS queries sent once per mode, one request at a time, after
20 warm-up requests per mode; k = 10; client and server on the same machine
over HTTP (uvicorn, 1 worker). Hardware: Intel Core i7-12650H laptop CPU
(10 cores / 16 threads), 16 GB RAM, Windows 11, Python 3.13, no GPU.

| Mode | Mean | p50 | p95 | p99 | Server time (mean) |
|---|---|---|---|---|---|
| BM25 | 2.6 ms | 2.4 ms | 4.4 ms | 5.2 ms | 0.7 ms |
| Dense | 17.9 ms | 17.7 ms | 19.4 ms | 21.1 ms | 14.3 ms |
| Hybrid (RRF) | 19.2 ms | 18.9 ms | 22.4 ms | 24.7 ms | 15.7 ms |
| Hybrid + LambdaRank | 24.5 ms | 23.9 ms | 27.5 ms | 30.6 ms | 21.3 ms |

Embedding the query with MiniLM on the CPU is most of the cost (about 13 ms);
retrieval, features for ~160 candidates and the 174-tree model add about 8 ms.
These are single-client numbers; throughput under concurrent load was not
measured.

## Docker

On a machine with only Git and Docker:

```bash
git clone https://github.com/Himanshuy1718/hybrid-search-engine.git
cd hybrid-search-engine
docker build -t hybrid-search .
docker run --rm -p 8000:8000 hybrid-search          # API  -> http://localhost:8000
docker run --rm -p 8501:8501 hybrid-search demo     # demo -> http://localhost:8501
```

(If port 8000 is taken, map another host port: `-p 8001:8000`.)

The build needs nothing from the local machine: `.dockerignore` excludes
`data/` and `artifacts/`, and the image rebuilds them itself. The first build
took about 15 minutes on the laptop below, about 9 of them embedding the
catalogue. Stages:

1. **base**: `python:3.13-slim` pinned by digest, CPU-only PyTorch (the
   default Linux wheel adds ~2 GB of CUDA libraries), pinned requirements.
2. **test**: the unit tests run and must pass, or the image is not built.
3. **pipeline**: downloads WANDS from a pinned commit and checks SHA-256
   hashes, downloads the model, then runs BM25 -> dense -> hybrid -> re-ranker.
   Only offline modules are copied in, so editing the API or demo never
   re-runs the pipeline.
4. **final**: data, embeddings, model, fusion weight and code. Runs as a
   non-root user, never contacts Hugging Face (`HF_HUB_OFFLINE=1`), and has a
   health check for either the API or the demo.

The image is about 2.3 GB of layers, mostly PyTorch (0.8 GB) and the
scientific Python stack.

**Results reproduce on Linux.** The build log prints the results table. BM25,
dense and hybrid match the Windows run to all four decimals. The re-ranker
scores NDCG@10 0.783 in the image vs 0.786 on Windows: the Linux and Windows
PyTorch builds produce embeddings that differ by at most 2e-7 (float32
rounding, in 93% of values). That changes no dense ranking, but it shifts the
cosine feature's histogram bins in LightGBM enough to change the trees chosen
per fold. The 0.003 gap is well inside the fold-to-fold spread (0.024).

**Thread limits matter in containers.** With default settings, PyTorch,
FAISS and LightGBM each start 8-16 threads, and in Docker Desktop's VM waking
them for one small query cost more than the work: re-ranked search took 116 ms
mean. The image sets `OMP_NUM_THREADS=2` (measured against 1, 2 and 4):

| Mode (in Docker) | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| BM25 | 3.5 ms | 3.2 ms | 5.3 ms | 6.2 ms |
| Dense | 17.9 ms | 17.7 ms | 19.7 ms | 20.9 ms |
| Hybrid (RRF) | 19.1 ms | 18.9 ms | 22.3 ms | 24.2 ms |
| Hybrid + LambdaRank | 25.5 ms | 24.7 ms | 30.9 ms | 34.7 ms |

Same method as the native table; Docker Desktop (WSL 2) on the same laptop.

## Data notes

- 42,994 products, 480 queries, 233,448 raw labels (Exact 11%, Partial 63%, Irrelevant 26%).
  The files are tab-separated despite the `.csv` extension.
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

**Re-ranker** (`src/features.py`, `src/rerank.py`). Candidates are the union
of BM25's and dense's top 100 (160 per query on average). Each candidate gets
12 features: BM25 score and rank, name-only BM25 score, cosine similarity and
rank, share of query words in the product name and class (with crude plural
stripping), exact phrase in name, class vote (share of the first-stage top-10s
with the same product class), average rating, review count and query length.
The same feature code runs offline and in the API.

- **`query_class` is deliberately not used.** It matches the class of 64% of
  Exact products and 4% of Irrelevant ones, but it describes the judged
  products and would not exist for a new query, so it would inflate offline
  scores with information the live system can't have. Category match is
  computed from query-time information only (class overlap, class vote).
- LightGBM `lambdarank`, rows grouped by query, `label_gain = [0, 1, 2]` so
  training optimises the same linear gains the evaluation uses (LightGBM's
  default is 2^gain - 1).
- Per outer fold, the number of trees is picked by early stopping on one of
  the four *training* folds, then the model is refit on all four. Other
  settings are fixed, not tuned. A test scrambles the test fold's labels and
  checks its predictions don't change.
- The deployed model is trained on all queries with the median tree count
  from CV (174), saved to `artifacts/models/lambdarank.txt`.

## Results

Mean over the 379 scored queries; in brackets, the standard deviation of the
five per-fold means.

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.704 (0.038) | 0.675 (0.058) | 0.620 (0.033) |
| Dense | 0.703 (0.019) | 0.621 (0.034) | 0.580 (0.033) |
| Hybrid (RRF) | 0.753 (0.018) | 0.686 (0.043) | 0.623 (0.045) |
| Hybrid + LambdaRank | **0.786** (0.024) | **0.712** (0.068) | **0.662** (0.040) |

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
- **The re-ranker beats the hybrid in every fold** (by 0.022 to 0.040
  NDCG@10); overall +0.033 (95% CI +0.022 to +0.045; better on 184 queries,
  worse on 81; p = 3e-10) and +0.083 over BM25. Its Recall@100 is higher than
  the hybrid's because it picks its top 100 from the larger union pool; it can
  never exceed that pool's 0.715.

NDCG@10 per fold:

| Fold | BM25 | Dense | Hybrid | + LambdaRank |
|---|---|---|---|---|
| 0 | 0.683 | 0.726 | 0.756 | 0.789 |
| 1 | 0.719 | 0.684 | 0.746 | 0.785 |
| 2 | 0.762 | 0.720 | 0.782 | 0.822 |
| 3 | 0.670 | 0.693 | 0.735 | 0.757 |
| 4 | 0.679 | 0.688 | 0.746 | 0.777 |

### What the re-ranker relies on

Split-gain importance (share of total gain, mean over the five fold models):

| Feature | Gain % | | Feature | Gain % |
|---|---|---|---|---|
| bm25_rank | 28.8 | | bm25_score | 3.9 |
| name_overlap | 22.1 | | cosine | 3.7 |
| dense_rank | 20.0 | | review_count | 2.5 |
| class_vote | 8.6 | | class_overlap | 2.0 |
| bm25_name_score | 6.6 | | query_len, average_rating, name_phrase | < 1 each |

Gain importance splits credit between correlated features (BM25 rank vs
score) and understates features that help through interactions, so the
ablation below is the better guide. It re-runs the full 5-fold CV with one
feature group removed:

| Removed | NDCG@10 | Change |
|---|---|---|
| nothing (all features) | 0.786 | |
| text match (name/class overlap, phrase) | 0.772 | -0.015 |
| BM25 (score, rank, name score) | 0.773 | -0.013 |
| dense (cosine, rank) | 0.774 | -0.012 |
| popularity (rating, reviews) | 0.776 | -0.010 |
| class vote | 0.777 | -0.009 |

Every group helps and no single one carries the model; the groups partly
overlap, so removing one costs about 0.01. Popularity gets only 3% of the
gain but is worth as much as the dense features. These single-run
differences are not individually significance-tested.

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
