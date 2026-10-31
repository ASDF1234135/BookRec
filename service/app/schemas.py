from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: str | None = None
    recent_book_ids: list[str] | None = None
    k: int = Field(10, ge=1, le=50)


class Item(BaseModel):
    book_id: str
    title: str | None = None
    score: float


class RecommendResponse(BaseModel):
    items: list[Item]
    strategy: str
    model_version: str
    latency_ms: float


class BookResponse(BaseModel):
    book_id: str
    title: str | None = None
    authors: list[str] = []
    average_rating: float | None = None
