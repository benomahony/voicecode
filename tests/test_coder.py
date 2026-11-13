import asyncio
from pathlib import Path

from code_chat.index import Hit
from voicecode.coder import CodeEditor, describe, repository_of


def hit(file: Path, start: int, end: int) -> Hit:
    return {
        "file": str(file),
        "path": file.name,
        "start": start,
        "end": end,
        "end_col": 0,
        "symbols": [],
        "preview": "def retry():",
        "score": 0.9,
    }


def test_repository_is_the_git_root_above_the_buffer(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    assert repository_of(str(tmp_path / "src" / "a.py")) == tmp_path, "expected the git root"
    assert repository_of("untitled.py") is None, "unnamed buffers must not search the cwd"


def test_hits_show_their_code_under_a_location_header(tmp_path: Path) -> None:
    source = tmp_path / "dl.py"
    _ = source.write_text("import time\n\ndef with_retry(fetch, url):\n    return fetch(url)\n")
    shown = describe([hit(source, 3, 4)])
    assert shown == "dl.py:3-4\ndef with_retry(fetch, url):\n    return fetch(url)", shown


def test_deleted_files_fall_back_to_the_preview(tmp_path: Path) -> None:
    shown = describe([hit(tmp_path / "gone.py", 1, 2)])
    assert shown == "gone.py:1-2\ndef retry():", f"expected the preview, got {shown!r}"


def test_agent_searches_through_its_tool_and_carries_on_without_a_repository() -> None:
    output = asyncio.run(CodeEditor("test").transform("retry like the downloader", "untitled.py"))
    assert "search_code" in output, f"the test model calls every tool; got {output!r}"
    assert "isn't in a git repository" in output, f"expected the no-repository answer: {output!r}"
