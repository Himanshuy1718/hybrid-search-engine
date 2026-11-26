"""Entry point for Streamlit Community Cloud.

Deploy settings: repository Himanshuy1718/hybrid-search-engine, branch main,
main file path deploy/streamlit/streamlit_app.py, Python 3.13.

Community Cloud installs deploy/streamlit/requirements.txt (CPU-only PyTorch)
instead of the root requirements.txt, and runs from the repository root. On
first start the app downloads the data and trained artifacts
(src/cloud_setup.py) before loading the engine.
"""
import os
import runpy
import sys
from pathlib import Path

# One small query at a time: a few threads beat one per core (see README, Docker).
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
