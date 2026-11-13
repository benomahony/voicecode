import numpy as np

from voicecode.owner import BufferOwner
from voicecode.speech import downmix


class FakeEditor:
    def __init__(self, content: str = "", cursor: int = 0) -> None:
        self.buffers: dict[int, str] = {1: content}
        self.active: int = 1
        self.cursor: int = cursor
        self.style: tuple[str, str] = ("", "")

    def current(self) -> tuple[int, int]:
        return self.active, self.cursor

    def name(self, buffer: int) -> str:
        return f"{buffer}.py"

    def read(self, buffer: int) -> str:
        return self.buffers[buffer]

    def splice(self, buffer: int, span: range, text: str) -> None:
        content = self.buffers[buffer]
        self.buffers[buffer] = content[: span.start] + text + content[span.stop :]

    def comment(self, buffer: int) -> tuple[str, str]:
        assert buffer in self.buffers, f"no buffer {buffer}"
        return self.style

    def place_cursor(self, buffer: int, offset: int) -> None:
        if buffer == self.active:
            self.cursor = offset

    def type(self, offset: int, text: str) -> None:
        self.splice(self.active, range(offset, offset), text)


def test_speech_is_inserted_at_cursor_not_end_of_buffer() -> None:
    editor = FakeEditor("x = 1\n\ny = 2", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function")
    owner.partial("create a function named hello")
    assert editor.read(1) == "x = 1\ncreate a function named hello\ny = 2", (
        "speech must land on the cursor line, revised in place, not at the end of the buffer"
    )
    assert editor.cursor == len("x = 1\ncreate a function named hello"), (
        "the cursor must follow the speech so the next words continue it"
    )


def test_speech_follows_cursor_moves_without_duplicating_words() -> None:
    editor = FakeEditor("a\nb", cursor=1)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("one two")
    editor.cursor = 0
    owner.partial("one two three")
    assert editor.read(1) == "three a one two\nb", (
        "after a cursor move, placed words stay put and only new words go to the cursor"
    )


def test_speech_follows_active_buffer() -> None:
    editor = FakeEditor("first")
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("hello")
    editor.buffers[2] = ""
    editor.active, editor.cursor = 2, 0
    owner.partial("hello world")
    assert editor.read(1) == "hello first", (
        "words placed before a buffer switch must stay in the first buffer"
    )
    assert editor.read(2) == "world", (
        "speech after a buffer switch must go to the newly active buffer"
    )


def test_code_edit_lands_while_speech_continues() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function named hello")
    snapshot = owner.snapshot()
    assert snapshot is not None, "speech in the buffer must give a snapshot for the model"
    owner.partial("create a function named hello that returns")
    assert owner.apply_code(snapshot, "def hello():\n    pass\n"), (
        "a model edit must apply when only speech was appended since the snapshot"
    )
    assert editor.read(1) == "def hello():\n    pass that returns", (
        "speech that arrived during the model call must follow the generated code"
    )
    owner.partial("create a function named hello that returns world")
    assert editor.read(1) == "def hello():\n    pass that returns world", (
        "later partials must extend the leftover speech without repeating consumed words"
    )


def test_code_edit_survives_next_utterance_starting() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    snapshot = owner.snapshot()
    assert snapshot is not None, "speech in the buffer must give a snapshot for the model"
    owner.start_utterance()
    owner.partial("and y two")
    assert owner.apply_code(snapshot, "x = 1"), (
        "a model edit must still apply after the next utterance has started"
    )
    assert editor.read(1) == "x = 1 and y two", (
        "the new utterance must follow the code that replaced the previous one"
    )
    owner.partial("and y two please")
    assert editor.read(1) == "x = 1 and y two please", (
        "the new utterance must keep revising in place after the edit"
    )


def test_revised_speech_rejects_outdated_code_edit() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function")
    snapshot = owner.snapshot()
    assert snapshot is not None, "speech in the buffer must give a snapshot for the model"
    owner.partial("create the function")
    assert not owner.apply_code(snapshot, "def wrong(): pass"), (
        "a model edit must be rejected once the speech it saw was revised"
    )
    assert editor.read(1) == "create the function", (
        "a rejected edit must leave the revised speech untouched"
    )


def test_manual_edits_are_never_overwritten() -> None:
    editor = FakeEditor("x = 1\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("add a function")
    snapshot = owner.snapshot()
    assert snapshot is not None, "speech in the buffer must give a snapshot for the model"
    editor.type(0, "# note\n")
    assert not owner.apply_code(snapshot, "x = 3"), (
        "a model edit must be rejected after a manual edit"
    )
    assert editor.read(1) == "# note\nx = 1\nadd a function", (
        "a manual edit must survive a rejected model edit"
    )


def test_older_response_cannot_overwrite_newer_applied_edit() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    older = owner.snapshot()
    owner.partial("make x one and y two")
    newer = owner.snapshot()
    assert older is not None and newer is not None, (
        "both utterances must give snapshots for the model"
    )
    assert owner.apply_code(newer, "x = 1\ny = 2"), (
        f"the newer model edit must apply: snapshot {newer!r}, buffer {editor.read(1)!r}"
    )
    assert not owner.apply_code(older, "x = 1"), (
        "an older model response must never overwrite a newer applied edit"
    )
    assert editor.read(1) == "x = 1\ny = 2", "the newer edit must still be in the buffer"


def test_moved_cursor_is_left_alone_by_code_edit() -> None:
    editor = FakeEditor("z = 0\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    snapshot = owner.snapshot()
    assert snapshot is not None, "speech in the buffer must give a snapshot for the model"
    editor.cursor = 0
    assert owner.apply_code(snapshot, "z = 0\nx = 1"), (
        "a model edit must apply even though the user moved the cursor"
    )
    assert editor.cursor == 0, "a model edit must not drag the cursor back after the user moved it"
    owner.partial("make x one now")
    assert editor.read(1) == "now z = 0\nx = 1", (
        "speech after the edit must go to where the user moved the cursor"
    )


def test_downmix_hears_any_channel_of_an_aggregate_device() -> None:
    frames = np.zeros((4, 3), dtype=np.float32)
    frames[:, 2] = 0.5  # meeting audio on the third channel, mic silent
    frames[0, :] = 0.9  # everyone at once clips instead of wrapping
    mixed = downmix(frames)
    assert mixed.shape == (4,), "downmix must give one mono sample per frame"
    assert mixed.dtype == np.float32, "downmix must keep float32 samples for Whisper"
    assert list(mixed) == [1.0, 0.5, 0.5, 0.5], (
        "audio on any channel must be heard, and overlapping speakers clip rather than wrap"
    )


def python(content: str = "", cursor: int = 0) -> FakeEditor:
    editor = FakeEditor(content, cursor)
    editor.style = ("#", "")
    return editor


def test_speech_is_a_tagged_comment_in_the_buffer_syntax() -> None:
    editor = python("x = 1\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make a function")
    assert editor.read(1) == "x = 1\n# AI: make a function", editor.read(1)


def test_line_comment_mid_line_becomes_a_trailing_comment() -> None:
    editor = python("foo(bar)", cursor=4)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("pass a timeout")
    assert editor.read(1) == "foo(bar) # AI: pass a timeout", editor.read(1)


def test_block_comment_wraps_speech_and_speech_cannot_close_it() -> None:
    editor = FakeEditor("x;", cursor=0)
    editor.style = ("/*", "*/")
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("stop */ here")
    assert editor.read(1) == "/* AI: stop  here */ x;", editor.read(1)


def test_commented_speech_keeps_flowing_after_a_code_edit() -> None:
    editor = python()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    snapshot = owner.snapshot()
    assert snapshot is not None, "commented speech must give a snapshot"
    owner.partial("make x one and y two")
    assert owner.apply_code(snapshot, "x = 1"), f"edit must rebase over {editor.read(1)!r}"
    assert editor.read(1) == "x = 1 # AI: and y two", editor.read(1)
    owner.partial("make x one and y two three")
    assert editor.read(1) == "x = 1 # AI: and y two three", (
        f"later words must follow without dropping or repeating any: {editor.read(1)!r}"
    )


def test_cursor_move_counts_spoken_words_not_comment_markup() -> None:
    editor = python("a = 1\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("one two")
    editor.cursor = 0
    owner.partial("one two three")
    assert editor.read(1) == "a = 1 # AI: three\n# AI: one two", (
        f"the moved-to speech must hold exactly the new word: {editor.read(1)!r}"
    )


def test_speech_never_splits_the_word_under_the_cursor() -> None:
    editor = FakeEditor("int x;", cursor=1)
    editor.style = ("/*", "*/")
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make a function")
    assert editor.read(1) == "int /* AI: make a function */ x;", (
        f"speech must go after the word, not inside it: {editor.read(1)!r}"
    )


def test_indented_python_keeps_its_indentation() -> None:
    editor = python("def f():\n    return x", cursor=10)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("return y instead")
    assert editor.read(1) == "def f():\n    return x # AI: return y instead", editor.read(1)
