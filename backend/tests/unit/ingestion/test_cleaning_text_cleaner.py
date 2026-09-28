"""TextCleaner composition: configured order, signal merging, and title cleaning."""

import pytest

from chat_app.core.models import RawArticle
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.cleaning.models import CleaningContext, StepResult
from chat_app.ingestion.cleaning.text_cleaner import TextCleaner


class Recorder:
    def __init__(self, name: str, log: list[str], removes: str = ""):
        self.name = name
        self._log = log
        self._removes = removes

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        self._log.append(self.name)
        new = text.replace(self._removes, "") if self._removes else text
        removed = {self.name: len(text) - len(new)} if new != text else {}
        return StepResult(text=new, chars_removed_by_rule=removed, truncation_markers=[])


def raw(text: str, title: str = "T") -> RawArticle:
    return RawArticle(title=title, link=" https://x/1 ", ticker="AAPL", full_text=text)


def test_runs_steps_in_given_order_and_reports_per_step_removals():
    log: list[str] = []
    cleaner = TextCleaner(
        steps=[Recorder("first", log, "ab"), Recorder("second", log, "c")], title_steps=[]
    )
    result = cleaner.clean(raw("abcd"))
    assert log == ["first", "second"]
    assert result.text == "d"
    assert result.signals.chars_removed_by_step == {"first": 2, "second": 1}
    assert result.signals.chars_removed_by_rule == {"first": 2, "second": 1}
    assert result.link == "https://x/1"


def test_title_is_cleaned_with_title_steps(cleaner):
    result = cleaner.clean(raw("Body.", title="Apple’s Launch – Update "))
    assert result.title == "Apple's Launch - Update"


def test_unknown_step_in_config_is_rejected(cleaning_config):
    config = cleaning_config.model_copy(update={"step_order": ["whitespace", "nope"]})
    with pytest.raises(ValueError, match="nope"):
        build_text_cleaner(config, [])


def test_step_order_comes_from_config(cleaning_config):
    cleaner = build_text_cleaner(cleaning_config, [])
    assert [s.name for s in cleaner._steps] == cleaning_config.step_order
