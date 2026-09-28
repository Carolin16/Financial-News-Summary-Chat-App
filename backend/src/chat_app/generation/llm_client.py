"""OpenAI-backed LLM client: structured extraction (offline) and streamed answers (online).

Timeouts and retries with exponential backoff are delegated to the SDK, which already
retries connection errors, 408/409/429 and 5xx responses; callers handle the final failure.
"""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI, OpenAIError
from pydantic import BaseModel

from chat_app.config.settings import Settings
from chat_app.core.models import TokenUsage


class LlmError(RuntimeError):
    """Raised when the LLM cannot produce a usable response after retries."""


class OpenAILlmClient:
    """Thin adapter over the OpenAI Responses API."""

    def __init__(self, settings: Settings) -> None:
        """Create sync and async clients sharing the configured timeout and retry policy."""
        options = {
            "api_key": settings.openai_api_key.get_secret_value(),
            "base_url": settings.openai_base_url,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        self._model = settings.llm_model
        self._sync = OpenAI(**options)  # type: ignore[arg-type]
        self._async = AsyncOpenAI(**options)  # type: ignore[arg-type]

    def parse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Return the model's answer validated against `schema` (structured output)."""
        try:
            response = self._sync.responses.parse(
                model=self._model, instructions=instructions, input=prompt, text_format=schema
            )
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if response.output_parsed is None:
            raise LlmError("LLM returned no parseable structured output")
        return response.output_parsed

    async def aparse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Async variant of `parse` for use inside request handlers."""
        try:
            response = await self._async.responses.parse(
                model=self._model, instructions=instructions, input=prompt, text_format=schema
            )
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if response.output_parsed is None:
            raise LlmError("LLM returned no parseable structured output")
        return response.output_parsed

    async def stream_text(
        self, instructions: str, prompt: str, usage: TokenUsage
    ) -> AsyncIterator[str]:
        """Yield answer text deltas; token counts are written to `usage` when done."""
        try:
            async with self._async.responses.stream(
                model=self._model, instructions=instructions, input=prompt
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                final = await stream.get_final_response()
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if final.usage is not None:
            usage.input_tokens = final.usage.input_tokens
            usage.output_tokens = final.usage.output_tokens
