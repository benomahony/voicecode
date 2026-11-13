from collections.abc import Callable
from types import TracebackType
from typing import Literal, Protocol, Self, TypedDict, overload

import numpy as np
import numpy.typing as npt

class CallbackFlags(Protocol):
    def __bool__(self) -> bool: ...

class _InputStream(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

class DeviceInfo(TypedDict):
    name: str
    index: int
    max_input_channels: int
    default_samplerate: float

class _QueryDevices(Protocol):
    @overload
    def __call__(self) -> list[DeviceInfo]: ...
    @overload
    def __call__(self, device: int | str | None, kind: Literal["input"]) -> DeviceInfo: ...

query_devices: _QueryDevices

class _InputStreamClass(Protocol):
    def __call__(
        self,
        device: int,
        samplerate: float,
        channels: int,
        dtype: str,
        blocksize: int,
        callback: Callable[[npt.NDArray[np.float32], int, object, CallbackFlags], None],
    ) -> _InputStream: ...

InputStream: _InputStreamClass
