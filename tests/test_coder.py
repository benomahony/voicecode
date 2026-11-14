import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from voicecode.coder import (
    DIVIDER,
    REPLACE,
    SEARCH,
    Change,
    CodeEditor,
    Kind,
    apply,
    changes,
    echoed,
)

BUFFER = "def f():\n    # AI: return two\n    return 1\n\ndef g():\n    return 1"
PYTHON = Kind("python", ("#", ""))


def block(search: str, replace: str) -> str:
    return f"{SEARCH}\n{search}\n{DIVIDER}\n{replace}\n{REPLACE}"


def test_blocks_are_read_from_the_reply_in_order() -> None:
    reply = "Sure.\n" + block("a", "b") + "\n" + block("c\nd", "e")
    assert changes(reply) == [Change("a", "b"), Change("c\nd", "e")], changes(reply)


def test_only_the_changed_lines_are_replaced() -> None:
    edit = Change("    # AI: return two\n    return 1", "    return 2")
    out = apply(BUFFER, [edit])
    assert out == "def f():\n    return 2\n\ndef g():\n    return 1", repr(out)


def test_a_search_that_matches_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="2 times") as error:
        _ = apply(BUFFER, [Change("    return 1", "    return 2")])
    assert "return 1" in str(error.value), f"the error should quote the search: {error.value}"


def test_a_search_that_is_not_in_the_buffer_is_refused() -> None:
    with pytest.raises(ValueError, match="isn't in the buffer") as error:
        _ = apply(BUFFER, [Change("    return 3", "    return 2")])
    assert "return 3" in str(error.value), f"the error should quote the search: {error.value}"


def test_the_agent_makes_the_blocks_it_was_given() -> None:
    coder = CodeEditor("test")
    reply = TestModel(
        custom_output_text=block("    # AI: return two\n    return 1", "    return 2")
    )
    with coder.agent.override(model=reply):
        out = asyncio.run(coder.transform(BUFFER, "a.py", PYTHON))
    assert out.startswith("def f():\n    return 2\n"), f"the block must be applied: {out!r}"


def test_a_reply_without_blocks_changes_nothing_and_says_why() -> None:
    coder = CodeEditor("test")
    with coder.agent.override(model=TestModel(custom_output_text="I can't do that.")):
        out = asyncio.run(coder.transform(BUFFER, "a.py", PYTHON))
    assert out == "", f"prose must not become the buffer: {out!r}"
    assert coder.failure == "the model made no changes", (
        f"the log needs a reason: {coder.failure!r}"
    )


def test_echoed_lines_are_the_unchanged_ones_a_reply_repeats() -> None:
    edits = [Change("    a = 1\n    # AI: double b\n    b = 2", "    a = 1\n    b = 4")]
    assert echoed(edits) == ["    a = 1"], f"only the unchanged line is an echo: {echoed(edits)}"
