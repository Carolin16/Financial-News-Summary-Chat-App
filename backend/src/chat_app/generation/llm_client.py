"""Talks to the OpenAI API, with timeouts and automatic retries when a call fails."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, Omit, OpenAI, OpenAIError, omit
from pydantic import BaseModel

from chat_app.config.settings import Settings
from chat_app.core.models import TokenUsage


class LlmError(RuntimeError):
    """Raised when the LLM still gives no usable answer after all retries."""


class OpenAILlmClient:
    """A small wrapper so the rest of the app never calls OpenAI directly."""

    def __init__(
        self, settings: Settings, model: str | None = None, temperature: float | None = None
    ) -> None:
        """Set up the connection, using a different model or temperature if one is given."""
        options = {
            "api_key": settings.openai_api_key.get_secret_value(),
            "base_url": settings.openai_base_url,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        self._model = model or settings.llm_model
        # If no temperature is given, leave it out so the model uses its own default.
        self._temperature: float | Omit = omit if temperature is None else temperature
        self._sync = OpenAI(**options)  # type: ignore[arg-type]
        self._async = AsyncOpenAI(**options)  # type: ignore[arg-type]

    def parse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Ask the LLM and get its answer back in a fixed shape (e.g. a yes/no verdict)."""
        try:
            response = self._sync.responses.parse(
                model=self._model,
                instructions=instructions,
                input=prompt,
                text_format=schema,
                temperature=self._temperature,
            )
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if response.output_parsed is None:
            raise LlmError("LLM returned no parseable structured output")
        return response.output_parsed

    async def aparse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Same as `parse`, but without blocking the server while it waits."""
        try:
            response = await self._async.responses.parse(
                model=self._model,
                instructions=instructions,
                input=prompt,
                text_format=schema,
                temperature=self._temperature,
            )
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if response.output_parsed is None:
            raise LlmError("LLM returned no parseable structured output")
        return response.output_parsed

    async def stream_text(
        self, instructions: str, prompt: str, usage: TokenUsage
    ) -> AsyncIterator[str]:
        """Send the answer back piece by piece as it is written, then record tokens used."""
        try:
            async with self._async.responses.stream(
                model=self._model,
                instructions=instructions,
                input=prompt,
                temperature=self._temperature,
            ) as stream:
                # Pass on only the new text, ignoring the other status events.
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                final = await stream.get_final_response()
        except OpenAIError as error:
            raise LlmError(str(error)) from error
        if final.usage is not None:
            usage.input_tokens = final.usage.input_tokens
            usage.output_tokens = final.usage.output_tokens
