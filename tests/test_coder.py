import asyncio

from pydantic_ai.models.test import TestModel

from voicecode.coder import WINDOW_LINES, CodeEditor, resembles, unfenced, window

LINES = [f"line {n}" for n in range(100)]
CONTENT = "\n".join(LINES)


def test_the_window_is_whole_lines_around_the_speech() -> None:
    speech_line = 50
    start = CONTENT.index(f"line {speech_line}")
    region = window(CONTENT, range(start, start + 4))
    lines = CONTENT[region.start : region.stop].split("\n")
    assert lines[0] == f"line {speech_line - WINDOW_LINES}", f"window starts at {lines[0]!r}"
    assert lines[-1] == f"line {speech_line + WINDOW_LINES}", f"window ends at {lines[-1]!r}"


def test_the_window_stops_at_the_ends_of_the_buffer() -> None:
    region = window(CONTENT, range(4))
    assert region.start == 0, f"a window can't start before the buffer: {region}"
    end = window(CONTENT, range(len(CONTENT) - 4, len(CONTENT)))
    assert end.stop == len(CONTENT), f"a window can't end past the buffer: {end}"


def test_only_the_window_changes_and_the_rest_of_the_buffer_is_kept() -> None:
    content = CONTENT.replace("line 50", "# AI: make line 50 shout")
    start = content.index("# AI:")
    coder = CodeEditor("test")
    rewritten = TestModel(custom_output_text="line 49\nLINE 50\nline 51")
    with coder.agent.override(model=rewritten):
        out = asyncio.run(coder.transform(content, "a.py", range(start, start + 24)))
    lines = out.split("\n")
    assert lines[:30] == LINES[:30], f"lines above the window must be untouched: {lines[:30]}"
    assert lines[30:33] == ["line 49", "LINE 50", "line 51"], f"window not replaced: {lines[30:33]}"
    assert lines[33:] == LINES[71:], f"lines below the window must be untouched: {lines[33:]}"


def test_a_fenced_region_loses_its_fence() -> None:
    assert unfenced("```python\nx = 1\n```") == "x = 1", "the fence must not reach the buffer"
    assert unfenced("x = 1") == "x = 1", "an unfenced region must come back unchanged"


def test_prose_is_not_taken_for_the_region() -> None:
    region = "def f():\n    # AI: return two\n    return 1"
    assert not resembles("Sure! Here is the updated function.", region), (
        f"prose replaced {region!r}"
    )
    assert resembles("def f():\n    return 2", region), f"a rewrite of {region!r} shares lines"
