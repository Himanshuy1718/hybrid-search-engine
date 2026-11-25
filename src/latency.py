"""Measure API latency against a running server.

    uvicorn src.api:app --port 8000        (in another terminal)
    python -m src.latency                  (or --url http://host:port)

Sends each of the 480 WANDS queries once per mode, one request at a time,
after a warm-up of 20 requests per mode that are not counted. Reports
client-side wall time (HTTP round trip on the same machine) and the
server-side time spent in the engine.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from time import perf_counter

import httpx
import numpy as np
import pandas as pd

from src.data import load_queries
from src.engine import MODES
from src.evaluate import ARTIFACTS_DIR

WARMUP = 20


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware() -> dict:
    info = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version()}
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def measure(url: str, queries: list[str], mode: str, k: int = 10) -> pd.DataFrame:
    rows = []
    with httpx.Client(base_url=url, timeout=30) as client:
        for q in queries[:WARMUP]:
            client.get("/search", params={"q": q, "k": k, "mode": mode}).raise_for_status()
        for q in queries:
            t0 = perf_counter()
            r = client.get("/search", params={"q": q, "k": k, "mode": mode})
            wall = (perf_counter() - t0) * 1000
            r.raise_for_status()
            rows.append({"mode": mode, "wall_ms": wall, "server_ms": r.json()["took_ms"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("mode", sort=False)
    out = pd.DataFrame({
        "mean_ms": g["wall_ms"].mean(),
        "p50_ms": g["wall_ms"].median(),
        "p95_ms": g["wall_ms"].quantile(0.95),
        "p99_ms": g["wall_ms"].quantile(0.99),
        "server_mean_ms": g["server_ms"].mean(),
    })
    return out.round(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()

    httpx.get(f"{args.url}/health", timeout=10).raise_for_status()
    queries = load_queries()["query"].tolist()
    df = pd.concat([measure(args.url, queries, m) for m in args.modes], ignore_index=True)
    table = summarize(df)
    info = hardware()
    print(json.dumps(info, indent=2))
    print(f"\n{len(queries)} queries per mode, k=10, sequential, after {WARMUP} warm-up requests per mode:")
    print(table.to_string())

    out = ARTIFACTS_DIR / "eval" / "latency.json"
    out.write_text(json.dumps({"hardware": info, "queries_per_mode": len(queries),
                               "results": table.reset_index().to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
