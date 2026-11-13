from collections.abc import Callable
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt

class CallbackFlags:
    def __bool__(self) -> bool: ...

class InputStream:
    def __init__(
        self,
        samplerate: float,
        channels: int,
        dtype: str,
        blocksize: int,
        callback: Callable[[npt.NDArray[np.float32], int, object, CallbackFlags], None],
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
