"""GET /health: used by Docker and Compose healthchecks."""

import logging

from fastapi import APIRouter, Request, Response, status

from chat_app.api.container import Container
from chat_app.api.schemas import HealthResponse
from chat_app.retrieval.qdrant_schema import collection_ready

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    """200 while Qdrant is reachable (even before indexing); 503 when it is not."""
    container: Container = request.app.state.container
    try:
        ready = await collection_ready(container.qdrant, container.settings.qdrant_collection)
    except Exception:  # any client/transport failure means the dependency is down
        logger.warning("qdrant health check failed", exc_info=True)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", qdrant="down", index_ready=False)
    return HealthResponse(status="ok", qdrant="up", index_ready=ready)
