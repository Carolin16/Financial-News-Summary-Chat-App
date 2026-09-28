"""Prompt templates stored as Markdown files next to this module."""

from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


@cache
def load_prompt(name: str) -> str:
    """Return the template `<name>.md`; templates use `str.format` placeholders."""
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
