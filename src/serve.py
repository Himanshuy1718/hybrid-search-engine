"""Container entry point.

    python -m src.serve          # API on port 8000 (default)
    python -m src.serve demo     # Streamlit demo on port 8501
    python -m src.serve health   # exit 0 if the API or demo answers (Docker health check)
    python -m src.serve <cmd>    # anything else is run as given, e.g. `pytest -q`
"""
from __future__ import annotations

import os
import sys

COMMANDS = {
    "api": ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
    "demo": ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0",
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
}


HEALTH_URLS = ("http://127.0.0.1:8000/health", "http://127.0.0.1:8501/_stcore/health")


def command(argv: list[str]) -> list[str]:
    if not argv:
        return COMMANDS["api"]
    return COMMANDS.get(argv[0], argv)


def healthy() -> bool:
    """True if the API or the demo answers its health check (Docker HEALTHCHECK)."""
    import urllib.request

    for url in HEALTH_URLS:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except OSError:
            continue
    return False


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["health"]:
        sys.exit(0 if healthy() else 1)
    cmd = command(args)
    os.execvp(cmd[0], cmd)
