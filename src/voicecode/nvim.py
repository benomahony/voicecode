"""Neovim as an Editor for BufferOwner, over pynvim's synchronous client."""


def char_to_position(text: str, offset: int) -> tuple[int, int]:
    """Character offset to Neovim's zero-based (row, byte column)."""
    assert 0 <= offset <= len(text), (
        f"offset {offset} is outside the {len(text)}-character buffer; "
        "convert offsets of the text they index"
    )
    lines = text[:offset].split("\n")
    row, col = len(lines) - 1, len(lines[-1].encode())
    assert row <= text.count("\n"), f"row {row} is past the last line of the buffer"
    return row, col


class NvimEditor:
    def __init__(self, address: str) -> None:
        import pynvim

        assert address, "no Neovim socket given; the plugin passes --socket from v:servername"
        self.nvim: pynvim.Nvim = pynvim.attach("socket", path=address)
        assert self.current()[0] > 0, f"attached to {address} but Neovim has no current buffer"

    def current(self) -> tuple[int, int]:
        window = self.nvim.current.window
        lines: list[str] = window.buffer[:]
        row, col = window.cursor
        assert 1 <= row <= len(lines), f"Neovim reported cursor row {row} in {len(lines)} lines"
        line = lines[row - 1]
        assert 0 <= col <= len(line.encode()), (
            f"Neovim reported byte column {col} on a {len(line.encode())}-byte line"
        )
        chars = len(line.encode()[:col].decode(errors="ignore"))
        if not self.nvim.api.get_mode()["mode"].startswith("i") and line:
            chars = min(chars + 1, len(line))  # Normal mode inserts after the cursor, like `a`.
        return window.buffer.number, sum(len(x) + 1 for x in lines[: row - 1]) + chars

    def name(self, buffer: int) -> str:
        assert buffer > 0, f"buffer {buffer} does not exist; Neovim numbers buffers from 1"
        name = self.nvim.buffers[buffer].name
        assert "\n" not in name, f"buffer {buffer} is named {name!r}; the prompt needs one line"
        return name or "untitled.py"

    def read(self, buffer: int) -> str:
        assert buffer > 0, f"buffer {buffer} does not exist; Neovim numbers buffers from 1"
        lines = self.nvim.buffers[buffer][:]
        assert lines, f"buffer {buffer} has no lines; Neovim buffers always have at least one"
        return "\n".join(lines)

    def splice(self, buffer: int, span: range, text: str) -> None:
        content = self.read(buffer)
        assert 0 <= span.start <= span.stop <= len(content), (
            f"splice {span} is outside buffer {buffer} ({len(content)} characters); "
            "the owner's copy of the buffer is stale"
        )
        (r1, c1), (r2, c2) = (char_to_position(content, i) for i in (span.start, span.stop))
        assert (r1, c1) <= (r2, c2), f"splice starts at {(r1, c1)}, after its end {(r2, c2)}"
        self.nvim.api.buf_set_text(self.nvim.buffers[buffer], r1, c1, r2, c2, text.split("\n"))

    def comment(self, buffer: int) -> tuple[str, str]:
        """The buffer's 'commentstring' split around %s, e.g. ("#", "") or ("/*", "*/");
        ("", "") when it has none, so speech is plain text."""
        assert buffer > 0, f"buffer {buffer} does not exist; Neovim numbers buffers from 1"
        style = self.nvim.api.get_option_value("commentstring", {"buf": buffer})
        if "%s" not in style:
            return ("", "")
        opener, closer = style.split("%s", 1)
        assert opener.strip(), f"buffer {buffer} has commentstring {style!r} with no opener"
        return opener.strip(), closer.strip()

    def place_cursor(self, buffer: int, offset: int) -> None:
        assert buffer > 0, f"buffer {buffer} does not exist; Neovim numbers buffers from 1"
        window = self.nvim.current.window
        if window.buffer.number != buffer:
            return
        content = self.read(buffer)
        assert 0 <= offset <= len(content), (
            f"cursor offset {offset} is outside buffer {buffer} ({len(content)} characters)"
        )
        normal = not self.nvim.api.get_mode()["mode"].startswith("i")
        if normal and offset and content[offset - 1] != "\n":
            offset -= 1
        row, col = char_to_position(content, offset)
        window.cursor = (row + 1, col)
