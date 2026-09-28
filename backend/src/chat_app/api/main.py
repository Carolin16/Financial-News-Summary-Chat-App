"""FastAPI application: `uvicorn chat_app.api.main:app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat_app.api.container import Container
from chat_app.api.routes import chat, health
from chat_app.config.logging import configure_logging
from chat_app.config.settings import Settings, get_settings


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app; tests inject settings or a prepared container."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container(settings)
        yield
        await app.state.container.close()

    app = FastAPI(title="Financial News Chat", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
