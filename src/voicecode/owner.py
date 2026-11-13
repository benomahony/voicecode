"""The single writer of speech and code edits into a buffer.

Speech is ordinary buffer text anchored at the cursor, written as a comment in the
buffer's own syntax (from 'commentstring') so linters don't flag it while it waits for
the model. Model edits are rebased over speech appended since they were requested and
rejected on any other change.
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
    comment: Callable[[int], tuple[str, str]]  # A buffer's comment opener and closer.


MARK = "AI:"  # Tags spoken comments, so the model can tell them from real ones.


def line_rest(text: str, offset: int) -> str:
    """What follows `offset` on its line."""
    assert 0 <= offset <= len(text), f"offset {offset} is outside the {len(text)}-character text"
    end = text.find("\n", offset)
    rest = text[offset:] if end < 0 else text[offset:end]
    assert "\n" not in rest, f"line_rest returned more than one line: {rest!r}"
    return rest


def word_end(text: str, offset: int) -> int:
    """`offset`, moved past the rest of a word it falls inside, so speech never splits one."""
    assert 0 <= offset <= len(text), f"offset {offset} is outside the {len(text)}-character text"
    end = offset
    if end and (text[end - 1].isalnum() or text[end - 1] == "_"):
        while end < len(text) and (text[end].isalnum() or text[end] == "_"):
            end += 1
    assert end >= offset, f"word_end moved {offset} backwards to {end}"
    return end


def anchor_offset(text: str, cursor: int, comment: tuple[str, str]) -> int:
    """Where speech for a cursor goes: at the cursor, but never inside a word; and for
    line comments with code after the cursor, at the end of that line as a trailing
    comment, since a line comment would otherwise comment out or split that code."""
    opener, closer = comment
    if opener and not closer and line_rest(text, cursor).strip():
        offset = cursor + len(line_rest(text, cursor))
    else:
        offset = word_end(text, cursor)
    assert cursor <= offset <= len(text), f"anchor {offset} is before cursor {cursor}"
    assert not line_rest(text, offset).strip() or not opener or closer or offset == cursor, (
        "a line comment must not be anchored before code on its line"
    )
    return offset


def line_before(text: str, offset: int) -> str:
    """What precedes `offset` on its line."""
    assert 0 <= offset <= len(text), f"offset {offset} is outside the {len(text)}-character text"
    head = text[text.rfind("\n", 0, offset) + 1 : offset]
    assert "\n" not in head, f"line_before returned more than one line: {head!r}"
    return head


def body(words: tuple[str, ...], comment: tuple[str, str], continuing: bool) -> str:
    """The spoken words as a voice comment, or bare when there's no comment syntax or they
    continue a voice comment already open on their line."""
    assert words, "body() needs words; render() returns nothing for silence"
    opener, closer = comment
    spoken = " ".join(words)
    if closer:
        spoken = spoken.replace(closer, "")  # Speech can't end the comment early.
    if not opener or continuing:
        return spoken
    text = f"{opener} {MARK} {spoken}" + (f" {closer}" if closer else "")
    assert text.startswith(opener), f"voice comment {text!r} must open with {opener!r}"
    return text


def render(words: tuple[str, ...], before: str, after: str, comment: tuple[str, str]) -> str:
    """Spoken words as buffer text between `before` and `after`, the rest of their line on
    each side. On a line whose voice comment is already open before them (a line comment),
    they continue it instead of opening another. A new line comment followed by code on
    its line ends the line, so that code stays code."""
    assert "\n" not in before, f"before is the start of one line, got {before!r}"
    assert "\n" not in after, f"after is the rest of one line, got {after!r}"
    if not words:
        return ""
    opener, closer = comment
    line_comment = bool(opener) and not closer
    continuing = line_comment and f"{opener} {MARK}" in before
    text = body(words, comment, continuing)
    if before[-1:] and not before[-1:].isspace():
        text = " " + text
    if line_comment and not continuing and after.strip():
        return text + "\n"
    if after[:1] and not after[:1].isspace():
        return text + " "
    return text


@dataclass
class Anchor:
    """Speech is ordinary buffer text at buffer[start:start + len(text)]."""

    buffer: int
    start: int
    comment: tuple[str, str] = ("", "")  # The buffer's comment opener and closer.
    text: str = ""
    words: tuple[str, ...] = ()  # The ASR words `text` renders.
    skip: int = 0  # ASR words of this utterance already committed as ordinary text


@dataclass(frozen=True)
class Snapshot:
    buffer: int
    content: str
    start: int  # Start of the speech region the model was shown.
    end: int  # End of the speech region the model was shown.
    words: tuple[str, ...]  # The ASR words in that region.


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
        comment = editor.comment(buffer)
        self.anchor: Anchor = Anchor(buffer, anchor_offset(content, offset, comment), comment)
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
        buffer, cursor = self.editor.current()
        content = self.editor.read(buffer)
        comment = self.editor.comment(buffer)
        offset = anchor_offset(content, cursor, comment)
        assert buffer > 0, f"editor reported buffer {buffer}; Neovim numbers buffers from 1"
        assert 0 <= offset <= len(content), (
            f"cursor offset {offset} is outside buffer {buffer} ({len(content)} characters); "
            "editor.current() must count characters, not bytes"
        )
        self.anchor = Anchor(buffer, offset, comment, skip=self.anchor.skip)
        self.content = content
        self.mark = (buffer, cursor)

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
            self.anchor.words = ()
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
            a.skip += len(a.words)
            self._anchor_at_cursor()
            a = self.anchor
        assert a.skip >= 0, f"skipping {a.skip} words; skip counts words already placed"
        spoken = tuple(words.split()[a.skip :])
        old_end = a.start + len(a.text)
        before = line_before(self.content, a.start)
        text = render(spoken, before, line_rest(self.content, old_end), a.comment)
        a.text, a.words = text, spoken
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
        return Snapshot(a.buffer, self.content, a.start, end, a.words)

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
        if current[end:] != snap.content[snap.end :]:
            return False
        new_end = map_offset(snap.content, replacement, snap.end)
        if a.start == snap.start:
            # Same utterance: the speech may have grown, so compare words, not text.
            if current[: a.start] != snap.content[: snap.start]:
                return False
            seen = len(snap.words)  # Words the model acted on.
            if a.words[:seen] != snap.words:
                return False  # The recognizer revised words the model already acted on.
            left = a.words[seen:]
            before = line_before(replacement, new_end)
            tail = render(left, before, line_rest(replacement, new_end), a.comment)
            a.skip += seen
            a.words, a.text, a.start = left, tail, new_end
        elif a.start >= snap.end and current[: snap.end] == snap.content[: snap.end]:
            # A later utterance after the snapshot's speech. Render it again: the comment
            # it continued may now be code, and then it needs a voice comment of its own.
            head = replacement[:new_end] + current[snap.end : a.start]
            a.text = render(
                a.words, line_before(head, len(head)), line_rest(replacement, new_end), a.comment
            )
            tail = current[snap.end : a.start] + a.text
            a.start = len(head)
        else:
            return False
        moved = self.mark is None or self.editor.current() != self.mark
        result = replacement[:new_end] + tail + replacement[new_end:]
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
