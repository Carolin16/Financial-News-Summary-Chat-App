"""OpenAILlmClient request options (the SDK call is replaced; no network)."""

from types import SimpleNamespace

import pytest
from openai import Omit
from pydantic import BaseModel

from chat_app.config.settings import Settings
from chat_app.generation.llm_client import OpenAILlmClient


class Answer(BaseModel):
    ok: bool


class RecordingResponses:
    def __init__(self):
        self.kwargs: dict = {}

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_parsed=Answer(ok=True))


def client(**options) -> tuple[OpenAILlmClient, RecordingResponses]:
    llm = OpenAILlmClient(Settings(openai_api_key="test", llm_model="default-model"), **options)
    responses = RecordingResponses()
    llm._async = SimpleNamespace(responses=responses)  # type: ignore[assignment]
    return llm, responses


async def test_defaults_use_configured_model_and_omit_temperature():
    llm, responses = client()
    await llm.aparse("i", "p", Answer)
    assert responses.kwargs["model"] == "default-model"
    assert isinstance(responses.kwargs["temperature"], Omit)  # left out of the request


@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_overrides_are_sent_when_given(temperature):
    llm, responses = client(model="judge-model", temperature=temperature)
    await llm.aparse("i", "p", Answer)
    assert responses.kwargs["model"] == "judge-model"
    assert responses.kwargs["temperature"] == temperature
