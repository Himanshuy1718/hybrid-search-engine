"""Streamlit demo.

    streamlit run app.py

Searches the 42,994-product WANDS catalogue with any of the four stages and
can show BM25 side by side. If the query is one of the 480 WANDS queries,
each result shows its human relevance label.
"""
import random

import pandas as pd
import streamlit as st

from src.data import load_labels, load_queries
from src.engine import SearchEngine

MODE_LABELS = {
    "rerank": "Hybrid + LambdaRank",
    "hybrid": "Hybrid (RRF)",
    "dense": "Dense (embeddings)",
    "bm25": "BM25 (keywords)",
}
LABEL_COLOURS = {"Exact": "#c8e6c9", "Partial": "#fff3c4", "Irrelevant": "#ffcdd2"}

st.set_page_config(page_title="Hybrid product search", layout="wide")


@st.cache_resource(show_spinner="Loading indexes and models (about 30 seconds; a fresh server "
                                "first downloads the data and trained files, 1-2 minutes)...")
def get_engine() -> SearchEngine:
    from src.cloud_setup import ensure_ready

    ensure_ready()  # no-op when data/ and artifacts/ already exist
    return SearchEngine.load()


@st.cache_data
def get_judgments():
    queries, labels = load_queries(), load_labels()
    qid_of = {q.strip().lower(): int(i) for i, q in zip(queries["query_id"], queries["query"])}
    judged = {int(q): dict(zip(g["product_id"], g["label"])) for q, g in labels.groupby("query_id")}
    return qid_of, judged, sorted(queries["query"].tolist())


engine = get_engine()
qid_of, judged, examples = get_judgments()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Ranking", list(MODE_LABELS), format_func=MODE_LABELS.get)
    compare = st.checkbox("Show BM25 side by side", value=True, disabled=mode == "bm25")
    k = st.slider("Results", min_value=5, max_value=30, value=10)
    if st.button("Try a random WANDS query"):
        st.session_state["query"] = random.choice(examples)
    st.caption("Catalogue: 42,994 Wayfair products (WANDS). Models: BM25, "
               "all-MiniLM-L6-v2 + FAISS, reciprocal rank fusion, LightGBM LambdaRank.")

st.title("Hybrid product search")
query = st.text_input("Search the catalogue", key="query", placeholder="e.g. modern farmhouse dining table")

if query.strip():
    qid = qid_of.get(query.strip().lower())
    labels = judged.get(qid) if qid is not None else None
    if labels is not None:
        st.info("This is one of the 480 WANDS test queries, so results show the human relevance "
                "labels: Exact, Partial, Irrelevant, or unlabelled (counted as irrelevant in evaluation).")

    modes = [mode, "bm25"] if compare and mode != "bm25" else [mode]
    for column, m in zip(st.columns(len(modes)), modes):
        hits, timings = engine.search(query.strip(), k, m)
        table = pd.DataFrame(engine.describe(hits))
        with column:
            st.subheader(MODE_LABELS[m])
            if table.empty:
                st.write("No results.")
                continue
            caption = f"{timings['total']:.0f} ms on the server"
            if labels is not None:
                table["label"] = table["product_id"].map(labels).fillna("unlabelled")
                n_exact = (table["label"] == "Exact").sum()
                caption += f" · {n_exact} of {len(table)} results are Exact"
            st.caption(caption)
            shown = table.rename(columns={"product_class": "class", "average_rating": "rating",
                                          "review_count": "reviews"})
            columns = ["rank"] + (["label"] if labels is not None else []) + ["name", "class", "rating", "reviews"]
            styled = shown[columns].style.format({"rating": "{:.1f}"}, na_rep="–")
            if labels is not None:
                styled = styled.map(
                    lambda v: f"background-color: {LABEL_COLOURS[v]}; color: #1b1b1b" if v in LABEL_COLOURS else "",
                    subset=["label"])
            st.dataframe(styled, hide_index=True)
