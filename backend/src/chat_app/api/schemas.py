"""Request/response models validated at the API boundary."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from chat_app.config.settings import get_settings


class ChatRequest(BaseModel):
    """A user question."""

    question: str = Field(min_length=1, examples=["What's the latest news on Intel?"])

    @field_validator("question")
    @classmethod
    def _normalise(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question must not be blank")
        max_chars = get_settings().max_query_chars
        if len(value) > max_chars:
            raise ValueError(f"question must be at most {max_chars} characters")
        return value


class HealthResponse(BaseModel):
    """Liveness plus readiness of the vector index."""

    status: Literal["ok", "degraded"]
    qdrant: Literal["up", "down"]
    index_ready: bool
