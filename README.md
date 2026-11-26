# Hybrid Semantic Search and Ranking Engine

**Live demo: https://hybrid-search-engine-yiyeyak88hsdetz2qx5s7q.streamlit.app/**
(the app sleeps when idle; the first visit may take a minute to wake it)

A two-stage product search system on the [WANDS](https://github.com/wayfair/WANDS)
dataset (Wayfair: 42,994 products, 480 queries, 233k human relevance labels).
Keyword search (BM25) and embedding search (MiniLM + FAISS) retrieve
candidates, reciprocal rank fusion merges them, and a LightGBM LambdaRank
model re-ranks them. Each stage is measured against plain keyword search with
the same metric code, queries and query-level cross-validation.

```
query -> BM25 + embedding search (FAISS) -> merged candidates -> LightGBM LambdaRank -> top 10
```

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.704 | 0.675 | 0.620 |
| Dense (MiniLM + FAISS) | 0.703 | 0.621 | 0.580 |
| Hybrid (RRF) | 0.753 | 0.686 | 0.623 |
| **Hybrid + LambdaRank** | **0.786** | **0.712** | **0.662** |

- **+0.083 NDCG@10 over BM25** (+12% relative), better in all five folds.
- **25 ms** per re-ranked search on a laptop CPU (p95 28 ms), served by a
  FastAPI endpoint, with a Streamlit demo.
- **One `docker build`** rebuilds everything from the raw data and reproduces
  the results on Linux.

## Quick start (Docker)

On a machine with only Git and Docker:

```bash
git clone https://github.com/Himanshuy1718/hybrid-search-engine.git
cd hybrid-search-engine
docker build -t hybrid-search .                    # ~15 min: tests, data, model, full pipeline
docker run --rm -p 8000:8000 hybrid-search          # API  -> http://localhost:8000
docker run --rm -p 8501:8501 hybrid-search demo     # demo -> http://localhost:8501
```

If port 8000 is taken, map another host port, for example `-p 8001:8000`.

## Deployment

**Streamlit Community Cloud (free).** The demo runs from
`deploy/streamlit/streamlit_app.py`. Community Cloud installs
`deploy/streamlit/requirements.txt` (the same pins as the main requirements,
minus the API and test packages, with CPU-only PyTorch wheels for Python
3.12-3.14) and `packages.txt` (`libgomp1` for LightGBM). The repository holds
no data or trained files, so on first start `src/cloud_setup.py` downloads
WANDS (pinned commit, SHA-256 checked) and the trained artifacts from a pinned
revision of
[Himanshuy18/hybrid-search-artifacts](https://huggingface.co/Himanshuy18/hybrid-search-artifacts),
then loads as usual. Tested in a clean Debian 11 / Python 3.13 container that
installs with `uv` and starts with no data: it fetched the data and trained
files in about 30 seconds, then the MiniLM model, and served searches using
about 750 MB of memory.

To deploy a copy: share.streamlit.io -> Create app -> this repository, branch
`main`, main file `deploy/streamlit/streamlit_app.py`, and under Advanced
settings choose Python 3.13. Trained files are published with
`python scripts/publish_artifacts.py`.

**Hugging Face Docker Space.** `deploy/huggingface/` and
`scripts/deploy_hf_space.py` deploy the same demo as a Docker Space (tested
locally). Hugging Face now requires a PRO subscription for Docker Spaces.

## Local setup

Python 3.13 (tested on Windows 11 and, in Docker, Debian).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.download          # WANDS CSVs from a pinned commit into data/, SHA-256 checked
python -m src.data              # data summary
pytest                          # 85 tests
python -m src.bm25              # BM25 baseline
python -m src.dense             # embeds products once (9-19 min on a laptop CPU), then dense retrieval
python -m src.hybrid            # reciprocal rank fusion, weight tuned per fold
python -m src.rerank            # candidate features + LambdaRank, 5-fold CV, final model
python -m src.rerank ablation   # CV score with each feature group removed

uvicorn src.api:app --port 8000     # API; http://localhost:8000 opens the interactive docs
streamlit run app.py                # demo on http://localhost:8501
python -m src.latency               # latency benchmark against the running API
```

Each stage writes its top-100 run to `artifacts/runs/`, per-query metrics to
`artifacts/eval/` and its summary row to `artifacts/results.csv`.

## Results

Mean over the 379 scored queries; in brackets, the standard deviation of the
five per-fold means.

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.704 (0.038) | 0.675 (0.058) | 0.620 (0.033) |
| Dense | 0.703 (0.019) | 0.621 (0.034) | 0.580 (0.033) |
| Hybrid (RRF) | 0.753 (0.018) | 0.686 (0.043) | 0.623 (0.045) |
| Hybrid + LambdaRank | **0.786** (0.024) | **0.712** (0.068) | **0.662** (0.040) |

- **Hybrid beats BM25** by +0.050 NDCG@10 (95% bootstrap CI +0.035 to
  +0.065; better on 188 queries, worse on 102; Wilcoxon p = 2e-10).
- **Dense alone ties BM25** (difference -0.001, CI -0.022 to +0.020) but wins
  on different queries: of 25,470 Exact products, 6,211 are in both top-100s,
  3,481 only in BM25's and 2,924 only in dense's. That is why fusion helps.
- **Dense makes softer mistakes**: its top 10 have fewer Irrelevant products
  (3% vs 6%) but fewer Exact ones (42% vs 44%). It finds the right kind of
  product more often than the exact one asked for, hence its lower MRR.
- **The re-ranker beats the hybrid in every fold** (by 0.022 to 0.040);
  overall +0.033 (95% CI +0.022 to +0.045; better on 184 queries, worse on 81;
  p = 3e-10). Its candidates are the union of both top-100 lists, which hold
  71% of each query's Exact products against 62% for the hybrid's top 100, so
  its Recall@100 is higher too; it can never exceed that pool's 0.715.

| Fold | BM25 | Dense | Hybrid | + LambdaRank |
|---|---|---|---|---|
| 0 | 0.683 | 0.726 | 0.756 | 0.789 |
| 1 | 0.719 | 0.684 | 0.746 | 0.785 |
| 2 | 0.762 | 0.720 | 0.782 | 0.822 |
| 3 | 0.670 | 0.693 | 0.735 | 0.757 |
| 4 | 0.679 | 0.688 | 0.746 | 0.777 |

Sanity check: a random ranking scores NDCG@10 0.007. NDCG@10 is fairly high
even for BM25 partly because Partial products earn gain 1 and make up most
labels (BM25's top 10: 44% Exact, 36% Partial, 6% Irrelevant, 14% unlabelled).

### What the re-ranker relies on

Split-gain importance (share of total gain, mean over the five fold models):

| Feature | Gain % | | Feature | Gain % |
|---|---|---|---|---|
| bm25_rank | 28.8 | | bm25_score | 3.9 |
| name_overlap | 22.1 | | cosine | 3.7 |
| dense_rank | 20.0 | | review_count | 2.5 |
| class_vote | 8.6 | | class_overlap | 2.0 |
| bm25_name_score | 6.6 | | query_len, average_rating, name_phrase | < 1 each |

Gain importance splits credit between correlated features and understates
features that help through interactions, so the ablation is the better guide.
It re-runs the full 5-fold CV with one feature group removed:

| Removed | NDCG@10 | Change |
|---|---|---|
| nothing (all features) | 0.786 | |
| text match (name/class overlap, phrase) | 0.772 | -0.015 |
| BM25 (score, rank, name score) | 0.773 | -0.013 |
| dense (cosine, rank) | 0.774 | -0.012 |
| popularity (rating, reviews) | 0.776 | -0.010 |
| class vote | 0.777 | -0.009 |

Every group helps and none carries the model alone. Popularity gets only 3%
of the gain but is worth about as much as the dense features. Differences
this small are within the fold-to-fold spread and are not individually
significance-tested.

## Failure analysis

The 20 queries with the lowest re-ranker NDCG@10 (out-of-fold) were read by
hand: the query, the top results with their labels, and the Exact products it
should have found. Each got one main cause, and each cause was then checked
across all 379 scored queries.

| Cause | Of 20 | Examples: what came back (what was wanted) |
|---|---|---|
| **Wrong product type**: descriptive words outweigh the product noun | 6 | "bathroom vanity knobs" -> vanities (knobs); "drum picture" -> picture frames (drum wall art); "coffee bar buffet" -> coffee tables and mugs (buffets); "70s inspired furniture" -> 70s-print bedding; "small wardrobe grey" -> garment racks (armoires); "toddler couch fold out" -> toddler beds |
| **Words the catalogue writes differently** | 4 | typos "foutains with brick look", "small loving roomtables"; "midcentury tv unit" (the catalogue mostly writes "mid century" or "mid-century": "midcentury" appears 42 times, "century" about 1,300); "outdoor waterproof chest" (the catalogue says "deck box") |
| **Plausible results that are unlabelled** | 4 | "rug for teen room" (3 labels in total; result #1 "teen pink area rug" is unlabelled); "white abstract" (2 labels); "living room ideas" and "patio design" (top results are living room sets and patio chairs, unlabelled) |
| **Constraints ignored**: size, brand, purpose | 3 | "46 inch closet door" -> 46'' shower doors; "liberty hardware francisco" (the brand appears only in some descriptions); "end table between recliners" |
| **Re-ranker overruled good retrieval** | 1 | "outdoor light fixtures": hybrid 0.70 -> re-ranker 0.09 |
| **Ambiguous query or questionable labels** | 2 | "rack glass" (Exact products include a shoe rack); "milk cow chair" (Exact includes a plain slipper chair) |

Checked across all queries:

- **Wrong product type is the largest real error.** On average 81% of the
  re-ranker's top 10 belong to a product class that has at least one Exact
  product for the query; for the 20 worst queries, 31%. The 58 queries below
  50% average NDCG@10 0.52, against 0.83 for the rest. (This check uses the
  labels, so it is analysis only, never a feature.)
- **Vocabulary mismatch**: 15 queries contain a word found in no product text:
  typos ("tjat", "desins"), glued tokens ("7qt"; the catalogue writes "7 qt"),
  and brand or model names the catalogue doesn't contain ("dorel",
  "w003096809"). They average 0.63 against 0.79.
- **Missing labels**: NDCG@10 and the share of unlabelled results in the top
  10 are strongly correlated (r = -0.63). Part of that is real failure (bad
  results often fall outside the judged pool), but spot checks find clearly
  relevant unlabelled results, so part of the measured error is label
  coverage, not ranking. The two queries with fewer than 20 labels score 0
  with every stage.
- **Re-ranker regressions**: it beats the hybrid by more than 0.2 NDCG@10 on
  27 queries and loses by more than 0.2 on 10. For "outdoor light fixtures"
  the light bulbs it ranked first were BM25 ranks 7-13 ("light" repeated in
  their descriptions) but dense ranks 232-510; the actual fixtures had median
  BM25 rank 204 and dense rank 45. With BM25 rank as its strongest feature,
  the model sided with BM25 when the two retrievers strongly disagreed.

## What I would try next

Ordered by how many of the failures above each would address:

1. **Product-type matching.** Predict the query's product class (a classifier
   trained on training-fold queries, or the nearest classes in embedding
   space) and add "candidate is in the predicted class" as a feature. Add a
   head-noun feature too: most queries end with the product noun ("bathroom
   vanity *knobs*"), so check whether that word is in the product's name or
   class. Targets the 58 queries with under 50% correct-class results.
2. **Spelling and compound normalisation** against the catalogue vocabulary
   (edit distance; "7qt" -> "7 qt", "midcentury" -> "mid century") and synonym
   expansion (couch/sofa, chest/deck box), or character n-gram matching.
3. **A cross-encoder on the top ~30 candidates** (for example
   `cross-encoder/ms-marco-MiniLM-L-6-v2`). It reads the query and product
   together, so it can tell the product noun from its modifiers. Its CPU
   latency would need measuring.
4. **Retriever-disagreement features** such as |BM25 rank - dense rank|, so
   the model can learn when one retriever is unreliable (the "outdoor light
   fixtures" case).
5. **Structured attributes**: index `product_features` (dimensions, finish,
   colour, style) and add match features for numbers with units and colours.
6. **Fine-tune the embedding model** on training-fold labels, separately per
   fold to avoid leakage.
7. **Measure label coverage**: judge the unlabelled top-10 results of the
   worst queries (by hand, or with an LLM judge checked against the human
   labels) to separate ranking errors from missing labels.

## Design choices

**BM25 is implemented directly** (`src/bm25.py`) instead of with the
`rank-bm25` package. That package's IDF goes negative for words in more than
half the documents and is then floored to a constant, so a filler word in 90%
of products ("the", "with") gets *more* weight than a real word in 50%. This
implementation uses the Lucene/Elasticsearch IDF,
`ln(1 + (N - n + 0.5) / (n + 0.5))`, which is always positive and always
favours rarer words. It precomputes a sparse product-by-term weight matrix,
so a query is a sum of a few columns (about 1.5 ms). Settings: k1 = 1.2,
b = 0.75 (Lucene defaults, not tuned); product name, class and description
joined; lowercase alphanumeric tokens, no stemming or stopword removal.

**Dense retrieval** (`src/dense.py`) embeds "name. class. description" with
`all-MiniLM-L6-v2` (384 dimensions, up to 256 word pieces, so the name and
class always fit). Vectors are normalised, so FAISS `IndexFlatIP` returns
exact cosine-similarity neighbours. At 43k products exact search takes a few
milliseconds, so an approximate index (HNSW, IVF) would add error for no gain.

**Hybrid** (`src/hybrid.py`) uses weighted reciprocal rank fusion,
`w / (60 + rank_bm25) + (1 - w) / (60 + rank_dense)`. RRF uses ranks only, so
BM25 scores and cosine similarities never need a common scale. The weight `w`
is tuned **per fold on the other four folds** (grid 0.0-1.0) and applied to
the held-out fold; every fold chose 0.4 or 0.5. The API uses the weight picked
on all scored queries (0.4).

**Re-ranker** (`src/features.py`, `src/rerank.py`). Candidates are the union
of BM25's and dense's top 100 (160 per query on average). Each gets 12
features: BM25 score and rank, name-only BM25 score, cosine similarity and
rank, share of query words in the product name and class (with crude plural
stripping), exact phrase in name, class vote (share of the first-stage top-10s
with the same product class), average rating, review count and query length.

- **`query_class` is deliberately not used.** It matches the class of 64% of
  Exact products and 4% of Irrelevant ones, but it describes the judged
  products and would not exist for a new query, so it would inflate offline
  scores with information the live system can't have. Category match uses
  query-time information only (class overlap, class vote).
- LightGBM `lambdarank` with rows grouped by query. LambdaRank weights each
  pair of products by how much NDCG would change if they swapped, so it
  concentrates on the top of the list, unlike a classifier that scores each
  product on its own. `label_gain = [0, 1, 2]` matches the evaluation's linear
  gains (LightGBM's default is 2^gain - 1).
- Per outer fold, the number of trees is picked by early stopping on one of
  the four *training* folds, then the model is refit on all four. Other
  settings are fixed. A test scrambles the test fold's labels and checks its
  predictions don't change.
- The deployed model is trained on all queries with the median tree count
  from CV (174 trees).

**Serving** (`src/engine.py`). The API and the demo share one engine that
loads everything once at startup (about 30 s) and reuses the offline feature
code, so the model sees identical features in training and serving. Online
fusion uses a plain-Python RRF that a test checks is identical to the offline
pandas version (the pandas group-by alone cost ~8 ms per query).

## Evaluation rules

Fixed before any model was built; they apply to every stage.

| Rule | Choice |
|---|---|
| Relevance gains | Exact = 2, Partial = 1, Irrelevant = 0 |
| Unlabelled products | Count as gain 0. WANDS labels only a sample of products per query, so some "misses" may really be relevant; absolute scores are conservative. Comparisons between stages stay fair because every stage is scored the same way. |
| "Relevant" for MRR and Recall | Exact only. Partial does not count. |
| NDCG | NDCG@10, linear gain, log2 discount. The ideal ranking uses all labelled products for the query, not only the retrieved ones. |
| MRR | MRR@10: 1 / position of the first Exact product in the top 10, else 0. |
| Recall | Recall@100: share of the query's Exact products found in the top 100. |
| Query set | Queries with at least one Exact product (MRR and Recall are undefined otherwise). The same queries and the same metric code (`src/metrics.py`) score every stage. |
| Duplicate labels | Majority vote per (query, product); ties go to the lower gain. |
| Splits | 5-fold cross-validation **by query**, never by query-product pair, so no test query is seen in training. Folds are fixed with a seed and cover all 480 queries. |
| Tuning | Fusion weight and tree count are tuned on training folds only. |

## Data notes

- 42,994 products, 480 queries, 233,448 raw labels (Exact 11%, Partial 63%,
  Irrelevant 26%). The files are tab-separated despite the `.csv` extension.
- 1,467 (query, product) pairs are labelled more than once; only 14 disagree.
  Resolved by majority vote, ties to the lower gain -> 231,873 labels.
- 101 queries have no Exact product, so 379 queries are scored. All 480 are
  in the folds, and the unscored ones are still used for training.
- Some queries have hundreds of Exact products (max 876), so Recall@100 cannot
  reach 1 for them.
- 14% of products have no description and 6.6% no product class; 22% have no
  ratings (`average_rating` is left missing rather than set to 0).
- 72 descriptions contain mis-encoded characters from the source file; left as is.

## API and latency

`GET /search?q=modern farmhouse dining table&k=5` (mode defaults to `rerank`;
also `hybrid`, `dense`, `bm25`). Also `GET /health`; `/` redirects to the
interactive docs.

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

The demo searches with any stage, can show BM25 side by side and, when the
query is one of the 480 WANDS queries, colours each result by its human label.

Latency: each of the 480 WANDS queries sent once per mode, one request at a
time, after 20 warm-up requests per mode; k = 10; client and server on the
same machine over HTTP (uvicorn, 1 worker). Hardware: Intel Core i7-12650H
laptop CPU (10 cores / 16 threads), 16 GB RAM, Windows 11, Python 3.13, no GPU.

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

## Docker details

The build needs nothing from the local machine: `.dockerignore` excludes
`data/` and `artifacts/`, and the image rebuilds them. The first build took
about 15 minutes on the laptop above, about 9 of them embedding the catalogue.

1. **base**: `python:3.13-slim` pinned by digest, CPU-only PyTorch (the
   default Linux wheel adds ~2 GB of CUDA libraries), pinned requirements.
2. **test**: the unit tests must pass, or the image is not built.
3. **pipeline**: downloads WANDS from a pinned commit with SHA-256 checks,
   downloads the model, then runs BM25 -> dense -> hybrid -> re-ranker. Only
   offline modules are copied in, so editing the API or demo never re-runs it.
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

**Thread limits matter in containers.** By default PyTorch, FAISS and LightGBM
each start 8-16 threads, and in Docker Desktop's VM waking them for one small
query cost more than the work: re-ranked search took 116 ms mean. The image
sets `OMP_NUM_THREADS=2` (measured against 1, 2 and 4):

| Mode (in Docker) | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| BM25 | 3.5 ms | 3.2 ms | 5.3 ms | 6.2 ms |
| Dense | 17.9 ms | 17.7 ms | 19.7 ms | 20.9 ms |
| Hybrid (RRF) | 19.1 ms | 18.9 ms | 22.3 ms | 24.2 ms |
| Hybrid + LambdaRank | 25.5 ms | 24.7 ms | 30.9 ms | 34.7 ms |

## Limitations

- **Unlabelled products count as irrelevant.** WANDS judges a sample of
  products per query. The effect is similar across stages (14% of BM25's and
  15% of dense's top 10 are unlabelled), but it lowers absolute scores and,
  as the failure analysis shows, makes some good results look like failures.
- **Small evaluation set.** 379 scored queries; the fold-to-fold standard
  deviation of NDCG@10 is about 0.02, so differences under ~0.01 (such as
  between ablation rows) are not reliable.
- **No behavioural data.** No query logs, clicks or purchases; relevance comes
  only from the WANDS labels, and the queries are WANDS's, not real traffic.
- **Product data gaps.** WANDS has no brand field, and the structured
  attributes in `product_features` are not indexed.
- **Mostly default settings.** Only the fusion weight and tree count are
  tuned; BM25 k1/b, the RRF constant and other LightGBM settings are defaults.
- **Scale.** An in-memory BM25 matrix and exact FAISS search suit 43k
  products. Millions would call for an approximate index (HNSW or IVF) and a
  search engine such as Elasticsearch or OpenSearch for keyword retrieval.
- **Latency** is single-client on one laptop; throughput under concurrent load
  was not measured.
- **Platform variation.** Re-ranker scores differ by about 0.003 NDCG@10
  between Windows and Linux builds (see Docker details).

## Project layout

```
src/
  data.py        loading, cleaning, duplicate labels, query folds
  download.py    pinned, checksummed WANDS download
  metrics.py     NDCG@10, MRR@10, Recall@100
  evaluate.py    shared harness: same queries, metrics and folds for every stage
  bm25.py        sparse BM25 with Lucene IDF
  dense.py       MiniLM embeddings + FAISS
  hybrid.py      weighted RRF, weight tuned per fold
  features.py    candidate generation + 12 re-ranker features (offline and online)
  rerank.py      LightGBM LambdaRank: CV, importance, ablation, final model
  engine.py      online search engine shared by the API and demo
  api.py         FastAPI service
  latency.py     latency benchmark
  serve.py       container entry point (api | demo | health)
  cloud_setup.py downloads data and trained files on a fresh deployment
app.py           Streamlit demo
deploy/
  streamlit/     Streamlit Community Cloud entry point and requirements
  huggingface/   Hugging Face Docker Space Dockerfile and card
scripts/         publish_artifacts.py, deploy_hf_space.py
tests/           85 unit tests (metrics, BM25, fusion, features, leakage, API, deployment)
Dockerfile       multi-stage build: test -> pipeline -> final image
packages.txt     Debian packages for Streamlit Community Cloud
```
