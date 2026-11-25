"""API checks with a fake engine (no indexes or models loaded)."""
import pytest
from fastapi.testclient import TestClient

from src.api import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def search(self, query, k, mode):
        self.calls.append((query, k, mode))
        hits = [(7, 2.5), (3, 1.0)][:k]
        return hits, {"encode_query": 1.0, "total": 3.25}

    def describe(self, hits):
        return [{"rank": i, "product_id": pid, "name": f"product {pid}", "product_class": "Desks",
                 "score": s, "average_rating": None, "review_count": 0}
                for i, (pid, s) in enumerate(hits, start=1)]


@pytest.fixture()
def client_and_engine():
    engine = FakeEngine()
    with TestClient(create_app(load_engine=lambda: engine)) as client:
        yield client, engine


def test_health(client_and_engine):
    client, _ = client_and_engine
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_results_and_defaults(client_and_engine):
    client, engine = client_and_engine
    r = client.get("/search", params={"q": "  oak desk  "})
    assert r.status_code == 200
    body = r.json()
    assert engine.calls == [("oak desk", 10, "rerank")]  # trimmed query, default k and mode
    assert body["took_ms"] == 3.25
    assert [h["product_id"] for h in body["results"]] == [7, 3]
    assert body["results"][0]["average_rating"] is None


def test_mode_and_k_are_passed_through(client_and_engine):
    client, engine = client_and_engine
    client.get("/search", params={"q": "desk", "k": 1, "mode": "bm25"})
    assert engine.calls[-1] == ("desk", 1, "bm25")


@pytest.mark.parametrize("params", [
    {"q": ""}, {"q": "   "}, {"q": "desk", "k": 0}, {"q": "desk", "k": 101},
    {"q": "desk", "mode": "magic"}, {}, {"q": "x" * 201},
])
def test_bad_requests_are_rejected(client_and_engine, params):
    client, engine = client_and_engine
    assert client.get("/search", params=params).status_code == 422
    assert engine.calls == []
