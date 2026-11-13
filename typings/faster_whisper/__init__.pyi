from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

class Segment:
    text: str

class TranscriptionInfo:
    language: str

class WhisperModel:
    def __init__(
        self, model_size_or_path: str, device: str = ..., compute_type: str = ...
    ) -> None: ...
    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        language: str | None = ...,
        beam_size: int = ...,
        condition_on_previous_text: bool = ...,
        vad_filter: bool = ...,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
