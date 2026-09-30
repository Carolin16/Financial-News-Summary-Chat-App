"""The /chat endpoint: takes a question and streams the answer back as it is written."""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette import EventSourceResponse

from chat_app.api.container import Container, VectorStoreUnavailableError
from chat_app.api.schemas import ChatRequest

router = APIRouter()
logger = logging.getLogger(__name__)

# Shown when the news database can't be reached.
SEARCH_UNAVAILABLE = "The news search index is unavailable right now. Please try again shortly."


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    """Answer a question, sending each step (sources, lines, final answer) as a live event."""
    container: Container = request.app.state.container
    try:
        service = await container.answer_service()
    except VectorStoreUnavailableError:
        logger.exception("vector store unavailable")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, SEARCH_UNAVAILABLE) from None

    async def events() -> AsyncIterator[dict[str, str]]:
        """Pass each answer event to the browser, stopping if the user leaves."""
        async for event in service.answer(body.question):
            if await request.is_disconnected():
                return
            yield {"event": event.type, "data": event.model_dump_json()}

    return EventSourceResponse(events())
