"""Builds what the LLM is told (rules, question, sources) and streams back its answer."""

from collections.abc import AsyncIterator

from chat_app.core.interfaces import StreamingLlm
from chat_app.core.models import SummaryRequest, TokenUsage
from chat_app.generation.prompts import load_prompt

# Prompt file names in prompts/: the shared rules, and the per-question-type guidance.
_SYSTEM_PROMPT = "answer_system"
_INTENT_PROMPT_PREFIX = "intent_"
_FALLBACK_INTENT = "news"


class LlmSummarizer:
    """Asks the LLM for an answer using the shared rules plus tips for this type of question."""

    def __init__(self, llm: StreamingLlm) -> None:
        """Take any LLM that can stream text, so it can be swapped or faked in tests."""
        self._llm = llm

    def build_instructions(self, request: SummaryRequest) -> str:
        """Write the LLM's instructions: fixed rules, question-type tips, and coverage warnings."""
        return load_prompt(_SYSTEM_PROMPT).format(
            intent_guidance=_intent_guidance(request.intent),
            coverage_guidance=request.coverage_guidance,
        )

    def build_prompt(self, request: SummaryRequest) -> str:
        """Write the message the LLM answers: the question, then the numbered sources."""
        return f"Question: {request.question}\n\nSources:\n\n{request.sources}"

    def stream(self, request: SummaryRequest, usage: TokenUsage) -> AsyncIterator[str]:
        """Send both to the LLM and pass its answer back piece by piece as it is written."""
        return self._llm.stream_text(
            self.build_instructions(request), self.build_prompt(request), usage
        )


def _intent_guidance(intent: str) -> str:
    """Load the tips for this question type, or the general news tips if it has none."""
    # So a new question type works right away and gets its own tips by adding a file.
    try:
        return load_prompt(f"{_INTENT_PROMPT_PREFIX}{intent}")
    except FileNotFoundError:
        return load_prompt(f"{_INTENT_PROMPT_PREFIX}{_FALLBACK_INTENT}")
