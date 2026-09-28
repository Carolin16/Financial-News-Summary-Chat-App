"""Prompt assembly for grounded summaries; the LLM transport is injected."""

from collections.abc import AsyncIterator

from chat_app.core.interfaces import StreamingLlm
from chat_app.core.models import SummaryRequest, TokenUsage
from chat_app.generation.prompts import load_prompt

_SYSTEM_PROMPT = "answer_system"
_INTENT_PROMPT_PREFIX = "intent_"
_FALLBACK_INTENT = "news"


class LlmSummarizer:
    """`Summarizer` that combines shared rules with per-intent guidance."""

    def __init__(self, llm: StreamingLlm) -> None:
        """Wrap any streaming LLM."""
        self._llm = llm

    def build_instructions(self, request: SummaryRequest) -> str:
        """System prompt: hard rules + guidance for this intent and coverage level."""
        return load_prompt(_SYSTEM_PROMPT).format(
            intent_guidance=_intent_guidance(request.intent),
            coverage_guidance=request.coverage_guidance,
        )

    def build_prompt(self, request: SummaryRequest) -> str:
        """User turn: the question followed by the numbered sources."""
        return f"Question: {request.question}\n\nSources:\n\n{request.sources}"

    def stream(self, request: SummaryRequest, usage: TokenUsage) -> AsyncIterator[str]:
        """Stream the answer text."""
        return self._llm.stream_text(
            self.build_instructions(request), self.build_prompt(request), usage
        )


def _intent_guidance(intent: str) -> str:
    # Intents without a dedicated template fall back to general news guidance (OCP: a new
    # intent works immediately and gains tailored guidance by adding a template file).
    try:
        return load_prompt(f"{_INTENT_PROMPT_PREFIX}{intent}")
    except FileNotFoundError:
        return load_prompt(f"{_INTENT_PROMPT_PREFIX}{_FALLBACK_INTENT}")
