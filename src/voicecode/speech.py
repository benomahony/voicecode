"""Microphone audio to words: channel mixing, the sounddevice callback and Whisper."""

import queue
import sys
from collections.abc import Callable
from typing import final

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.1
PARTIAL_INTERVAL = 0.8
SILENCE_SECONDS = 0.9
MIN_UTTERANCE_SECONDS = 0.4
MAX_UTTERANCE_SECONDS = 20.0
SILENCE_RMS = 0.012
BLOCKSIZE = 1600  # Samples per audio callback: FRAME_SECONDS at SAMPLE_RATE.


def downmix(indata: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Sum every input channel to mono, so aggregate devices (mic plus meeting audio) are heard."""
    assert indata.ndim == 2 and indata.shape[1] >= 1, (
        f"audio block has shape {indata.shape}; sounddevice delivers (frames, channels)"
    )
    mixed = np.clip(indata.sum(axis=1, dtype=np.float32), -1.0, 1.0)
    assert mixed.shape == (indata.shape[0],), f"mixed {indata.shape} into {mixed.shape}"
    return mixed


type Callback = Callable[[npt.NDArray[np.float32], int, object, object], None]


def audio_callback(frames: queue.SimpleQueue[npt.NDArray[np.float32]]) -> Callback:
    """The sounddevice callback: mix each block to mono and queue it for the asyncio loop."""
    assert frames.empty(), "start with an empty audio queue, or stale audio is transcribed"

    def on_audio(
        indata: npt.NDArray[np.float32], count: int, _time: object, status: object
    ) -> None:
        assert count == indata.shape[0], f"PortAudio said {count} frames, sent {indata.shape[0]}"
        assert count == BLOCKSIZE, f"got {count} frames per block; the stream asks for {BLOCKSIZE}"
        if status:
            print(f"Audio: {status}", file=sys.stderr, flush=True)
        frames.put(downmix(indata))

    assert BLOCKSIZE == round(FRAME_SECONDS * SAMPLE_RATE), (
        "BLOCKSIZE must hold FRAME_SECONDS of audio, or utterance timing drifts"
    )
    return on_audio


@final
class Transcriber:
    def __init__(self, model: str) -> None:
        from faster_whisper import WhisperModel

        assert model, "no Whisper model named; set --asr-model or VOICECODE_ASR_MODEL"
        assert model == model.strip(), (
            f"Whisper model {model!r} has surrounding whitespace; check VOICECODE_ASR_MODEL"
        )
        self.model = WhisperModel(model, device="auto", compute_type="auto")

    def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        assert audio.ndim == 1, f"audio has shape {audio.shape}; downmix() must make it mono"
        assert audio.size >= int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE), (
            f"{audio.size} samples is shorter than MIN_UTTERANCE_SECONDS; "
            "on_chunk must wait for enough speech"
        )
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
