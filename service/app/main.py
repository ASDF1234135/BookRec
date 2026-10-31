import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request

from .recommender import Recommender
from .schemas import BookResponse, RecommendRequest, RecommendResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bookrec")

app = FastAPI(title="bookrec")
rec = Recommender()


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    # One JSON line per request; Cloud Logging parses it so p95 and strategy mix are queryable.
    log.info(
        json.dumps(
            {
                "request_id": str(uuid.uuid4()),
                "path": request.url.path,
                "status": response.status_code,
                "strategy": getattr(request.state, "strategy", None),
                "k": getattr(request.state, "k", None),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "model_version": rec.model_version,
            }
        )
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok" if rec.loaded else "no_model", "model_version": rec.model_version}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    if not rec.loaded:
        raise HTTPException(503, "model not loaded")
    start = time.perf_counter()
    items, strategy = rec.recommend(req.user_id, req.recent_book_ids, req.k)
    request.state.strategy = strategy
    request.state.k = req.k
    return RecommendResponse(
        items=items,
        strategy=strategy,
        model_version=rec.model_version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: str):
    if not rec.loaded:
        raise HTTPException(503, "model not loaded")
    book = rec.book(book_id)
    if book is None:
        raise HTTPException(404, "book not found")
    return book
