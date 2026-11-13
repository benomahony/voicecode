import asyncio

from pydantic_ai.models.test import TestModel

from voicecode.coder import WINDOW_LINES, CodeEditor, window

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
    rewritten = TestModel(custom_output_args={"region": "LINE 50"})
    with coder.agent.override(model=rewritten):
        out = asyncio.run(coder.transform(content, "a.py", range(start, start + 24)))
    lines = out.split("\n")
    assert lines[:30] == LINES[:30], f"lines above the window must be untouched: {lines[:30]}"
    assert lines[30] == "LINE 50", (
        f"the window must be replaced by the model's region: {lines[30]!r}"
    )
    assert lines[31:] == LINES[71:], f"lines below the window must be untouched: {lines[31:]}"
