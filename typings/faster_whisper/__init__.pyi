from collections.abc import Iterable
from typing import Protocol

import numpy as np
import numpy.typing as npt

class Segment(Protocol):
    text: str

class TranscriptionInfo(Protocol):
    language: str

class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        language: str | None = ...,
        beam_size: int = ...,
        condition_on_previous_text: bool = ...,
        vad_filter: bool = ...,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...

class _WhisperModelClass(Protocol):
    def __call__(
        self, model_size_or_path: str, device: str = ..., compute_type: str = ...
    ) -> _WhisperModel: ...

WhisperModel: _WhisperModelClass
