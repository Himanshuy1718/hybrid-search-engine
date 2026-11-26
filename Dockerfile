# Build: docker build -t hybrid-search .
#        The first build took about 15 minutes on an 8-core laptop: it runs the
#        tests, downloads the data and the embedding model, and runs the whole
#        offline pipeline, including embedding 43k products on the CPU (~9 min).
#        Later builds reuse the cached layers; code-only changes take seconds.
# API:   docker run --rm -p 8000:8000 hybrid-search         -> http://localhost:8000
# Demo:  docker run --rm -p 8501:8501 hybrid-search demo    -> http://localhost:8501

# ---- base: Python and dependencies ------------------------------------------
# Pinned by digest so every build starts from the same image.
FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50 AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/opt/hf
# LightGBM needs the GNU OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch first, in its own layer: the default Linux wheel pulls in
# ~2 GB of CUDA libraries, and a separate layer survives a failed step below.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- test: unit tests must pass for the image to build -----------------------
FROM base AS test
COPY pyproject.toml ./
COPY deploy/streamlit/requirements.txt deploy/streamlit/requirements.txt
COPY src/ src/
COPY tests/ tests/
RUN python -m pytest -q && touch /tmp/tests-passed

# ---- pipeline: data -> BM25 -> embeddings -> fusion -> re-ranker -------------
# Only the offline modules are copied, so editing the API or demo does not
# re-run the pipeline.
FROM base AS pipeline
COPY src/__init__.py src/download.py src/data.py src/metrics.py src/evaluate.py \
     src/bm25.py src/dense.py src/hybrid.py src/features.py src/rerank.py src/
RUN python -m src.download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -m src.bm25 \
 && python -m src.dense \
 && python -m src.hybrid \
 && python -m src.rerank \
 && cat artifacts/results.csv

# ---- final: what runs -------------------------------------------------------
FROM base AS final
COPY --from=test /tmp/tests-passed /tmp/tests-passed
COPY --from=pipeline /opt/hf /opt/hf
COPY --from=pipeline /app/data data
COPY --from=pipeline /app/artifacts/embeddings artifacts/embeddings
COPY --from=pipeline /app/artifacts/models artifacts/models
COPY --from=pipeline /app/artifacts/hybrid_config.json /app/artifacts/results.csv artifacts/
COPY src/ src/
COPY app.py ./
RUN useradd --create-home --uid 10001 app
USER app
# Everything the container needs is inside it; never call Hugging Face at runtime.
ENV HF_HUB_OFFLINE=1
# PyTorch, FAISS and LightGBM each start a thread per core by default. For one
# small query at a time, waking 8-16 threads costs more than the work itself
# (re-ranked search: 116 ms mean with no limit, 25 ms with 2 threads, measured
# in Docker Desktop on the laptop below). Override with -e OMP_NUM_THREADS=n.
ENV OMP_NUM_THREADS=2
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-m", "src.serve", "health"]
ENTRYPOINT ["python", "-m", "src.serve"]
CMD ["api"]
