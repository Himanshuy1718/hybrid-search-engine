# Hugging Face Docker Space for the Streamlit demo.
#
# Built by Hugging Face from the folder that scripts/deploy_hf_space.py uploads.
# Unlike the main Dockerfile, it does not re-run the offline pipeline (about an
# hour on the Space's 2 CPUs): the trained artifacts (product embeddings,
# LambdaRank model, fusion weight) are uploaded with the code, and only the
# raw data and the MiniLM model are downloaded here.

FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/opt/hf
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY app.py ./
# WANDS from a pinned commit (SHA-256 checked) and the embedding model.
RUN python -m src.download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY artifacts/ artifacts/

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2
EXPOSE 8501
CMD ["python", "-m", "src.serve", "demo"]
