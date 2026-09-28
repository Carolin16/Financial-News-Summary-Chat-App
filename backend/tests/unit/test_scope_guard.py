from pydantic import BaseModel

from chat_app.generation.llm_client import LlmError
from chat_app.generation.scope_guard import LlmScopeGuard, ScopeVerdict


class FakeAsyncLlm:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def aparse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        self.calls.append((instructions, prompt, schema))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_returns_llm_verdict_and_sends_question_with_scope_rules():
    llm = FakeAsyncLlm(ScopeVerdict(in_scope=False))
    assert await LlmScopeGuard(llm).is_in_scope("Who is the president?") is False
    instructions, prompt, schema = llm.calls[0]
    assert "financial-news assistant" in instructions
    assert prompt == "Who is the president?"
    assert schema is ScopeVerdict


async def test_fails_open_when_llm_unavailable():
    assert await LlmScopeGuard(FakeAsyncLlm(LlmError("timeout"))).is_in_scope("x") is True
