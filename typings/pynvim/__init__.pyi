from typing import Literal, TypedDict, overload

class Mode(TypedDict):
    mode: str
    blocking: bool

class Buffer:
    number: int
    name: str
    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

class Window:
    buffer: Buffer
    cursor: tuple[int, int]

class Current:
    window: Window
    buffer: Buffer

class Buffers:
    def __getitem__(self, number: int) -> Buffer: ...

class Api:
    def get_mode(self) -> Mode: ...
    def buf_set_text(
        self,
        buffer: Buffer,
        start_row: int,
        start_col: int,
        end_row: int,
        end_col: int,
        replacement: list[str],
    ) -> None: ...

class Nvim:
    current: Current
    buffers: Buffers
    api: Api

def attach(session_type: Literal["socket"], *, path: str) -> Nvim: ...
