"""Acceptance: against the real model, an edit is a diff, not a rewrite.

Needs a model key, for example: OPENAI_API_KEY=... uv run pytest tests/test_acceptance.py
"""

import asyncio
import os

import pytest

from voicecode.coder import MODEL, CodeEditor, Kind, changes, echoed

CONTEXT = 2  # Unchanged lines a reply may repeat, to anchor its search.

UNTOUCHED = """def parse(line: str) -> dict[str, str]:
    key, _, value = line.partition("=")
    return {key.strip(): value.strip()}


def load(path: str) -> list[dict[str, str]]:
    with open(path) as handle:
        return [parse(line) for line in handle if line.strip()]


def describe(item: dict[str, str]) -> str:
    name = item.get("name", "unknown")
    price = item.get("price", "0")
    return f"{name}: {price}"
"""

EDITED = """def total(prices: list[float]) -> float:
    # AI: return the total in pence as an int instead of pounds
    return sum(prices)
"""

BUFFER = UNTOUCHED + "\n\n" + EDITED
PYTHON = Kind("python", ("#", ""))

needs_a_key = pytest.mark.skipif(
    os.environ.get("VOICECODE_MODEL", MODEL).startswith("openai:")
    and not os.environ.get("OPENAI_API_KEY"),
    reason="the acceptance test calls the real model; set OPENAI_API_KEY",
)


@needs_a_key
def test_the_model_sends_a_diff_not_a_rewrite_of_existing_code() -> None:
    coder = CodeEditor(os.environ.get("VOICECODE_MODEL", MODEL))
    result = asyncio.run(coder.transform(BUFFER, "shop.py", PYTHON))
    reply = coder.reply
    assert result, f"the edit must apply ({coder.failure}); the model replied:\n{reply}"
    assert "AI:" not in result, f"the spoken instruction must be replaced:\n{result}"
    assert result.startswith(UNTOUCHED), (
        f"code the speech didn't mention must not change:\n{result}"
    )
    repeated = echoed(changes(reply))
    assert len(repeated) <= CONTEXT, (
        f"the model repeated {len(repeated)} unchanged lines, more than {CONTEXT}: {repeated}"
    )


@needs_a_key
def test_the_end_of_a_sentence_extends_the_code_its_start_produced() -> None:
    """Speech arrives mid-sentence: the first edit acts on its start, then the rest arrives
    as a new AI: comment. With the file's conversation, the model must read the rest as the
    same request and extend that code, not drop it as a question."""
    coder = CodeEditor(os.environ.get("VOICECODE_MODEL", MODEL))
    start = "import typer\n\n# AI: build me a typer cli that tells me\n"
    first = asyncio.run(coder.transform(start, "cli.py", PYTHON))
    assert first, f"the start of the sentence must give an edit ({coder.failure}):\n{coder.reply}"
    rest = first.rstrip("\n") + "\n# AI: which models are available on the OpenAI API\n"
    second = asyncio.run(coder.transform(rest, "cli.py", PYTHON))
    assert second, f"the rest of the sentence must give an edit ({coder.failure}):\n{coder.reply}"
    assert "AI:" not in second, f"no part of the sentence may be left behind:\n{second}"
    assert "models" in second.lower(), f"the CLI must now list OpenAI models:\n{second}"
