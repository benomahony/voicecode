"""The single writer of speech and code edits into a buffer.

Speech is ordinary buffer text anchored at the cursor. Model edits are rebased over
speech appended since they were requested and rejected on any other change.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol


class Editor(Protocol):
    """Buffer text is lines joined by "\\n"; offsets are character indices into it.

    Members are declared as callables; any class with methods of these signatures fits.
    """

    current: Callable[[], tuple[int, int]]  # Active buffer and insertion offset at its cursor.
    name: Callable[[int], str]  # File name of a buffer, for the model's prompt.
    read: Callable[[int], str]  # Whole text of a buffer.
    splice: Callable[[int, range, str], None]  # Replace a character span of a buffer.
    place_cursor: Callable[[int, int], None]  # Move the cursor, if that buffer is active.


@dataclass
class Anchor:
    """Speech is ordinary buffer text at buffer[start:start + len(text)]."""

    buffer: int
    start: int
    text: str = ""
    skip: int = 0  # ASR words of this utterance already committed as ordinary text


@dataclass(frozen=True)
class Snapshot:
    buffer: int
    content: str
    end: int  # end of the speech region the model was shown


def map_offset(old: str, new: str, pos: int) -> int:
    """Where `pos` in `old` lands in `new`, after any change touching it."""
    assert 0 <= pos <= len(old), (
        f"offset {pos} is outside the {len(old)}-character original; map offsets of that text"
    )
    landed = len(new)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal" and i1 <= pos < i2:
            landed = j1 + pos - i1
            break
        if tag != "equal" and i1 <= pos <= i2:
            landed = j2
            break
    assert 0 <= landed <= len(new), (
        f"offset {pos} mapped to {landed}, outside the {len(new)}-character result; "
        "the opcodes must cover both texts"
    )
    return landed


class BufferOwner:
    """The only component allowed to mutate the Neovim buffer."""

    def __init__(self, editor: Editor) -> None:
        buffer, offset = editor.current()
        content = editor.read(buffer)
        assert buffer > 0, f"editor reported buffer {buffer}; Neovim numbers buffers from 1"
        assert 0 <= offset <= len(content), (
            f"cursor offset {offset} is outside buffer {buffer} ({len(content)} characters); "
            "editor.current() must count characters, not bytes"
        )
        self.editor: Editor = editor
        self.anchor: Anchor = Anchor(buffer, offset)
        self.content: str = content
        self.mark: tuple[int, int] | None = (
            None  # None: anchor at the cursor on the next utterance.
        )

    def _edited(self) -> bool:
        """Whether anything but this owner changed the anchored buffer since its last write."""
        assert self.anchor.buffer > 0, f"anchored to buffer {self.anchor.buffer}; anchor first"
        assert 0 <= self.anchor.start <= len(self.content), (
            f"anchor at {self.anchor.start} is outside the {len(self.content)} characters last "
            "written; move the anchor with every write"
        )
        return self.editor.read(self.anchor.buffer) != self.content

    def _diverged(self) -> bool:
        """Whether the user moved, switched buffer or edited since this owner last wrote."""
        where = self.editor.current()
        assert where[0] > 0 and where[1] >= 0, (
            f"editor reported cursor {where}; buffers count from 1 and offsets from 0"
        )
        assert self.mark is None or self.mark[0] > 0, (
            f"remembered cursor {self.mark} has no buffer; record it from editor.current()"
        )
        return self.mark is None or where != self.mark or self._edited()

    def _anchor_at_cursor(self) -> None:
        buffer, offset = self.editor.current()
        content = self.editor.read(buffer)
        assert buffer > 0, f"editor reported buffer {buffer}; Neovim numbers buffers from 1"
        assert 0 <= offset <= len(content), (
            f"cursor offset {offset} is outside buffer {buffer} ({len(content)} characters); "
            "editor.current() must count characters, not bytes"
        )
        self.anchor = Anchor(buffer, offset, skip=self.anchor.skip)
        self.content = content
        self.mark = (buffer, offset)

    def _write(self, span: range, text: str) -> None:
        assert 0 <= span.start <= span.stop <= len(self.content), (
            f"splice {span} is outside the {len(self.content)}-character buffer; "
            "compute spans from self.content"
        )
        expected = len(self.content) - len(span) + len(text)
        self.editor.splice(self.anchor.buffer, span, text)
        self.content = self.content[: span.start] + text + self.content[span.stop :]
        assert len(self.content) == expected, (
            f"buffer copy is {len(self.content)} characters after a splice, not {expected}"
        )

    def _follow(self) -> None:
        """Put the cursor after the speech and remember where we left it."""
        cursor = self.anchor.start + len(self.anchor.text)
        assert 0 <= cursor <= len(self.content), (
            f"cursor {cursor} would be outside the {len(self.content)}-character buffer; "
            "the anchor and its text must come from the last write"
        )
        self.editor.place_cursor(self.anchor.buffer, cursor)
        self.mark = self.editor.current()
        assert self.mark[0] > 0, f"editor reported cursor {self.mark}; buffers count from 1"

    def start_utterance(self) -> None:
        if self._diverged():
            self._anchor_at_cursor()
        else:
            self.anchor.start += len(self.anchor.text)
            self.anchor.text = ""
        self.anchor.skip = 0
        assert not self.anchor.text, "a new utterance must start with no speech placed"
        assert 0 <= self.anchor.start <= len(self.content), (
            f"utterance anchored at {self.anchor.start}, outside the "
            f"{len(self.content)}-character buffer; advance it by the placed speech only"
        )

    def partial(self, words: str) -> None:
        a = self.anchor
        if self._diverged():
            # The user moved or edited: placed speech stays, new speech follows the cursor.
            a.skip += len(a.text.split())
            self._anchor_at_cursor()
            a = self.anchor
        text = " ".join(words.split()[a.skip :])
        old_end = a.start + len(a.text)
        if text and a.start and not self.content[a.start - 1].isspace():
            text = " " + text
        if text and old_end < len(self.content) and not self.content[old_end].isspace():
            text += " "
        assert a.skip >= 0, f"skipping {a.skip} words; skip counts words already placed"
        a.text = text
        self._write(range(a.start, old_end), text)
        self._follow()
        assert self.content[a.start : a.start + len(text)] == text, (
            "placed speech is not where the anchor says; _write must splice at the anchor"
        )

    def snapshot(self) -> Snapshot | None:
        a = self.anchor
        if not a.text.strip() or self._edited():
            return None
        end = a.start + len(a.text)
        assert end <= len(self.content), (
            f"speech ends at {end}, past the {len(self.content)}-character buffer"
        )
        assert self.content[a.start : end] == a.text, (
            "anchored speech differs from the buffer; every speech write must go through _write"
        )
        return Snapshot(a.buffer, self.content, end)

    def apply_code(self, snap: Snapshot, replacement: str) -> bool:
        """Rebase a model edit onto speech appended since `snap`; reject anything else."""
        assert 0 <= snap.end <= len(snap.content), (
            f"snapshot speech ends at {snap.end}, outside its {len(snap.content)} characters; "
            "take snapshots from snapshot()"
        )
        a = self.anchor
        replacement = replacement.rstrip("\n")
        if not replacement.strip() or a.buffer != snap.buffer or self._edited():
            return False
        end = a.start + len(a.text)
        current = self.content
        if end < snap.end or current[: snap.end] != snap.content[: snap.end]:
            return False
        if current[end:] != snap.content[snap.end :]:
            return False
        moved = self.mark is None or self.editor.current() != self.mark
        new_end = map_offset(snap.content, replacement, snap.end)
        tail = current[snap.end : end]
        result = replacement[:new_end] + tail + replacement[new_end:]
        if a.start < snap.end:
            a.skip += len(current[a.start : snap.end].split())
            a.text = tail
            a.start = new_end
        else:
            a.start = new_end + a.start - snap.end
        prefix = len(os.path.commonprefix([current, result]))
        suffix = len(os.path.commonprefix([current[prefix:][::-1], result[prefix:][::-1]]))
        self._write(range(prefix, len(current) - suffix), result[prefix : len(result) - suffix])
        assert self.content == result, (
            "buffer after the code edit differs from the merged result; "
            "the common prefix and suffix must not overlap"
        )
        if not moved:
            self._follow()
        return True
