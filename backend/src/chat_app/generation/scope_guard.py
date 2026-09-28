"""Decides whether an ambiguous question is within the assistant's purpose.

Grounding checks prove a claim is *supported*; they can't tell that a question doesn't
*belong* here ("Who is the president?" is answerable from articles that mention him).
The analyzer's rules pass obvious financial questions straight through; only the rest
reach this guard, so the extra LLM call is paid for rare, unusual questions.
"""

import logging

from pydantic import BaseModel, Field

from chat_app.core.interfaces import AsyncStructuredLlm
from chat_app.generation.llm_client import LlmError
from chat_app.generation.prompts import load_prompt

logger = logging.getLogger(__name__)

_PROMPT_NAME = "scope_check"


class ScopeVerdict(BaseModel):
    """The guard's structured answer."""

    in_scope: bool = Field(description="True if the question is about business/financial news.")


class LlmScopeGuard:
    """LLM-backed scope classifier that fails open."""

    def __init__(self, llm: AsyncStructuredLlm) -> None:
        """Wrap any async structured-output LLM."""
        self._llm = llm

    async def is_in_scope(self, question: str) -> bool:
        """Return the verdict; on LLM failure allow the question through.

        Failing open is safe because answers are still grounded and verified; failing
        closed would block legitimate questions whenever the LLM is briefly unavailable.
        """
        try:
            verdict = await self._llm.aparse(load_prompt(_PROMPT_NAME), question, ScopeVerdict)
        except LlmError:
            logger.warning("scope check failed; allowing question", exc_info=True)
            return True
        return verdict.in_scope
