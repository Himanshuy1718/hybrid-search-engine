---
title: Hybrid Product Search
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: BM25 + embeddings + LambdaRank re-ranking on WANDS
---

# Hybrid product search

Search 42,994 Wayfair products (the [WANDS](https://github.com/wayfair/WANDS)
dataset) with four ranking stages and compare them side by side:

| Stage | NDCG@10 |
|---|---|
| BM25 (keywords) | 0.704 |
| Dense (all-MiniLM-L6-v2 + FAISS) | 0.703 |
| Hybrid (reciprocal rank fusion) | 0.753 |
| **Hybrid + LightGBM LambdaRank** | **0.786** |

Click **Try a random WANDS query** to see each result coloured by its human
relevance label (Exact, Partial, Irrelevant).

Code, evaluation method, failure analysis and limitations:
**https://github.com/Himanshuy1718/hybrid-search-engine**

The Space sleeps after two days without visitors; the first visit after that
takes a minute or two to wake it.
