import json

import pytest

from code_chat.daemon import LOG, read_results


def wire_hit(path: str, score: float) -> dict[str, object]:
    return {
        "file": f"/repo/{path}",
        "path": path,
        "start": 3,
        "end": 9,
        "end_col": 4,
        "symbols": ["retry"],
        "preview": "def retry():",
        "score": score,
    }


def test_results_skip_status_and_ready_messages() -> None:
    lines = [
        json.dumps({"type": "status", "text": "Loading EmbeddingGemma 2…"}),
        json.dumps({"type": "ready", "chunks": 2}),
        json.dumps(
            {
                "type": "results",
                "final": True,
                "seconds": 0,
                "hits": [wire_hit("a.py", 0.9), wire_hit("b.py", 0.5)],
            }
        ),
    ]
    hits = read_results(lines)
    assert [h["path"] for h in hits] == ["a.py", "b.py"], f"expected both hits in order, got {hits}"


def test_hang_up_before_results_is_an_error() -> None:
    with pytest.raises(ConnectionError, match="hung up") as error:
        _ = read_results([json.dumps({"type": "ready", "chunks": 2})])
    assert str(LOG) in str(error.value), f"the error should point at the daemon log: {error.value}"
