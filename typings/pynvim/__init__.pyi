from typing import Literal, Protocol, TypedDict, overload

class Mode(TypedDict):
    mode: str
    blocking: bool

class Buffer(Protocol):
    number: int
    name: str
    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

class Window(Protocol):
    buffer: Buffer
    cursor: tuple[int, int]

class Current(Protocol):
    window: Window
    buffer: Buffer

class Buffers(Protocol):
    def __getitem__(self, number: int) -> Buffer: ...

class Api(Protocol):
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

class Nvim(Protocol):
    current: Current
    buffers: Buffers
    api: Api

class _Attach(Protocol):
    def __call__(self, session_type: Literal["socket"], *, path: str) -> Nvim: ...

attach: _Attach
