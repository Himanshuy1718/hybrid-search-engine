"""FastAPI service.

    uvicorn src.api:app --port 8000
    GET /search?q=coffee+table&k=10            (mode=rerank by default)
    GET /search?q=coffee+table&k=10&mode=bm25  (also hybrid, dense)
    GET /health

All indexes and models are loaded once at startup (about 30 s), not per request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class Hit(BaseModel):
    rank: int
    product_id: int
    name: str
    product_class: str
    score: float
    average_rating: float | None
    review_count: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    k: int
    took_ms: float
    timings_ms: dict[str, float]
    results: list[Hit]


def create_app(load_engine=None) -> FastAPI:
    """load_engine lets tests pass a fake; by default the real engine is loaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine is None:
            from src.engine import SearchEngine

            app.state.engine = SearchEngine.load()
        else:
            app.state.engine = load_engine()
        yield

    app = FastAPI(title="Hybrid product search", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(..., min_length=1, max_length=200, description="search query"),
        k: int = Query(10, ge=1, le=100, description="number of results"),
        mode: Literal["rerank", "hybrid", "dense", "bm25"] = "rerank",
    ) -> SearchResponse:
        query = q.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query is empty")
        engine = request.app.state.engine
        hits, timings = engine.search(query, k, mode)
        return SearchResponse(
            query=query, mode=mode, k=k,
            took_ms=round(timings["total"], 2),
            timings_ms={step: round(ms, 2) for step, ms in timings.items()},
            results=engine.describe(hits),
        )

    return app


app = create_app()
