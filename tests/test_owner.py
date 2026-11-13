from voicecode.main import BufferOwner


class FakeEditor:
    def __init__(self, content: str = "", cursor: int = 0) -> None:
        self.buffers: dict[int, str] = {1: content}
        self.active: int = 1
        self.cursor: int = cursor

    def current(self) -> tuple[int, int]:
        return self.active, self.cursor

    def name(self, buffer: int) -> str:
        return f"{buffer}.py"

    def read(self, buffer: int) -> str:
        return self.buffers[buffer]

    def splice(self, buffer: int, start: int, end: int, text: str) -> None:
        content = self.buffers[buffer]
        self.buffers[buffer] = content[:start] + text + content[end:]

    def place_cursor(self, buffer: int, offset: int) -> None:
        if buffer == self.active:
            self.cursor = offset

    def type(self, offset: int, text: str) -> None:
        self.splice(self.active, offset, offset, text)


def test_speech_is_inserted_at_cursor_not_end_of_buffer() -> None:
    editor = FakeEditor("x = 1\n\ny = 2", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function")
    owner.partial("create a function named hello")
    assert editor.read(1) == "x = 1\ncreate a function named hello\ny = 2"
    assert editor.cursor == len("x = 1\ncreate a function named hello")


def test_speech_follows_cursor_moves_without_duplicating_words() -> None:
    editor = FakeEditor("a\nb", cursor=1)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("one two")
    editor.cursor = 0
    owner.partial("one two three")
    assert editor.read(1) == "three a one two\nb"


def test_speech_follows_active_buffer() -> None:
    editor = FakeEditor("first")
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("hello")
    editor.buffers[2] = ""
    editor.active, editor.cursor = 2, 0
    owner.partial("hello world")
    assert editor.read(1) == "hello first"
    assert editor.read(2) == "world"


def test_code_edit_lands_while_speech_continues() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function named hello")
    snapshot = owner.snapshot()
    assert snapshot is not None
    owner.partial("create a function named hello that returns")
    assert owner.apply_code(snapshot, "def hello():\n    pass\n")
    assert editor.read(1) == "def hello():\n    pass that returns"
    owner.partial("create a function named hello that returns world")
    assert editor.read(1) == "def hello():\n    pass that returns world"


def test_code_edit_survives_next_utterance_starting() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    snapshot = owner.finish_utterance()
    assert snapshot is not None
    owner.start_utterance()
    owner.partial("and y two")
    assert owner.apply_code(snapshot, "x = 1")
    assert editor.read(1) == "x = 1 and y two"
    owner.partial("and y two please")
    assert editor.read(1) == "x = 1 and y two please"


def test_revised_speech_rejects_outdated_code_edit() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("create a function")
    snapshot = owner.snapshot()
    assert snapshot is not None
    owner.partial("create the function")
    assert not owner.apply_code(snapshot, "def wrong(): pass")
    assert editor.read(1) == "create the function"


def test_manual_edits_are_never_overwritten() -> None:
    editor = FakeEditor("x = 1\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("add a function")
    snapshot = owner.snapshot()
    assert snapshot is not None
    editor.type(0, "# note\n")
    assert not owner.apply_code(snapshot, "x = 3")
    assert editor.read(1) == "# note\nx = 1\nadd a function"


def test_older_response_cannot_overwrite_newer_applied_edit() -> None:
    editor = FakeEditor()
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    older = owner.snapshot()
    owner.partial("make x one and y two")
    newer = owner.snapshot()
    assert older is not None and newer is not None
    assert owner.apply_code(newer, "x = 1\ny = 2")
    assert not owner.apply_code(older, "x = 1")
    assert editor.read(1) == "x = 1\ny = 2"


def test_moved_cursor_is_left_alone_by_code_edit() -> None:
    editor = FakeEditor("z = 0\n", cursor=6)
    owner = BufferOwner(editor)
    owner.start_utterance()
    owner.partial("make x one")
    snapshot = owner.snapshot()
    assert snapshot is not None
    editor.cursor = 0
    assert owner.apply_code(snapshot, "z = 0\nx = 1")
    assert editor.cursor == 0
    owner.partial("make x one now")
    assert editor.read(1) == "now z = 0\nx = 1"
