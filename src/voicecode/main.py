"""Live ASR and code editing with one owner of Neovim buffer mutations."""

from __future__ import annotations

import asyncio
import json
import math
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, final

import numpy as np
import numpy.typing as npt
import typer
from pydantic_ai.exceptions import ModelAuthenticationError, UserError
from pynvim.api import NvimError

from voicecode.coder import CodeEditor
from voicecode.nvim import NvimEditor
from voicecode.owner import BufferOwner
from voicecode.speech import (
    BLOCKSIZE,
    FRAME_SECONDS,
    MAX_UTTERANCE_SECONDS,
    MIN_UTTERANCE_SECONDS,
    PARTIAL_INTERVAL,
    SAMPLE_RATE,
    SILENCE_RMS,
    SILENCE_SECONDS,
    Transcriber,
    audio_callback,
)

LEVELS = ("info", "error", "auth")  # What the plugin's on_stdout understands.
WRITER_THREAD = "voicecode-writer"


def say(message: str, level: str = "info") -> None:
    """User-facing status for the Neovim plugin: one "level<TAB>message" line on stdout."""
    assert level in LEVELS, f"status level {level!r} is not one the plugin handles: {LEVELS}"
    assert message.strip(), "empty status message; the plugin would show a blank notification"
    print(f"{level}\t{' '.join(message.split())}", flush=True)


@final
@dataclass(frozen=True)
class Writer:
    """pynvim's sync client runs its own event loop, so it cannot be called from asyncio.
    One dedicated thread performs every Neovim call, which also serialises all writes."""

    pool: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix=WRITER_THREAD)
    )

    async def call[T](self, fn: Callable[[], T]) -> T:
        assert not threading.current_thread().name.startswith(WRITER_THREAD), (
            "Writer.call from the writer thread would deadlock; call the function directly"
        )
        assert asyncio.current_task() is not None, (
            "Writer.call must be awaited inside a task so its result has somewhere to go"
        )
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn)


@dataclass(frozen=True)
class Options:
    socket: str
    model: str
    asr_model: str
    device: str | None  # Input device name (or unique part of it); None is the system default.


@dataclass
class Session:
    """Microphone loop feeding the buffer owner and, continuously, the code model."""

    options: Options
    writer: Writer
    editor: NvimEditor
    owner: BufferOwner
    transcriber: Transcriber
    coder: CodeEditor
    frames: queue.SimpleQueue[npt.NDArray[np.float32]] = field(default_factory=queue.SimpleQueue)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    coding: asyncio.Task[None] | None = None
    rerun: bool = False
    last_sent: str = ""
    utterance: list[npt.NDArray[np.float32]] = field(default_factory=list)
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    speaking: bool = False
    last_voice: float = 0.0
    last_partial: float = 0.0
    serial: int = 0
    final_task: asyncio.Task[None] | None = None

    @classmethod
    async def run(cls, options: Options) -> None:
        """Connect to Neovim, load the models, and listen until the input device stops."""
        assert options.socket, "no Neovim socket given; the plugin passes --socket"
        assert options.device is None or options.device.strip(), (
            "--device is blank; leave it out to use the system default input"
        )
        coder = CodeEditor(options.model)  # Fails fast on a missing API key, before the ASR load.
        writer = Writer()
        editor = await writer.call(partial(NvimEditor, options.socket))
        owner = await writer.call(partial(BufferOwner, editor))
        say("Loading speech model")
        transcriber = await asyncio.to_thread(Transcriber, options.asr_model)
        await cls(options, writer, editor, owner, transcriber, coder).listen()

    async def edit_snapshot(self) -> None:
        assert self.coding is not None, "edit_snapshot runs only as the task transform_soon starts"
        try:
            snapshot = await self.writer.call(self.owner.snapshot)
            if snapshot is None or snapshot.content == self.last_sent:
                return
            assert snapshot.buffer > 0, (
                f"snapshot of buffer {snapshot.buffer}; buffers count from 1"
            )
            self.last_sent = snapshot.content
            filename = await self.writer.call(partial(self.editor.name, snapshot.buffer))
            speech = range(snapshot.start, snapshot.end)
            asked = time.monotonic()
            replacement = await self.coder.transform(snapshot.content, filename, speech)
            took = time.monotonic() - asked
            if await self.writer.call(partial(self.owner.apply_code, snapshot, replacement)):
                print(f"Edit applied; the model took {took:.1f}s", file=sys.stderr, flush=True)
            else:
                reason = self.owner.rejected
                print(f"Edit discarded after {took:.1f}s: {reason}", file=sys.stderr, flush=True)
        except ModelAuthenticationError as exc:
            say(f"The model provider rejected the API key: {exc}", "auth")
        except (RuntimeError, NvimError) as exc:
            say(f"Code edit failed: {exc}", "error")

    def transform_soon(self) -> None:
        """Keep one model call in flight; newer speech queues exactly one rerun."""
        assert self.coding is not None or not self.rerun, (
            "a rerun is queued with no model call running; transform_done must clear it"
        )
        if self.coding is not None:
            self.rerun = True
            return
        self.coding = asyncio.create_task(self.edit_snapshot())
        self.coding.add_done_callback(self.transform_done)
        assert not self.rerun, "a fresh model call already covers the latest speech"

    def transform_done(self, task: asyncio.Task[None]) -> None:
        assert task is self.coding, "a model call finished that transform_soon did not track"
        assert task.done(), "transform_done runs as a done callback, after the task finishes"
        self.coding = None
        if self.rerun:
            self.rerun = False
            self.transform_soon()

    async def recognise(self, audio: npt.NDArray[np.float32], sequence: int) -> None:
        """A newer ASR result supersedes an older one; only the owner edits."""
        assert 0 < sequence <= self.serial, (
            f"recognition {sequence} was never requested (latest is {self.serial})"
        )
        assert audio.ndim == 1, f"audio has shape {audio.shape}; downmix() must make it mono"
        try:
            words = await asyncio.to_thread(self.transcriber.transcribe, audio)
            if sequence != self.serial:
                return
            await self.writer.call(partial(self.owner.partial, words))
            self.transform_soon()
        except (RuntimeError, ValueError, NvimError) as exc:
            say(f"Transcription failed: {exc}", "error")

    async def begin_utterance(self) -> None:
        assert not self.speaking, "an utterance began during another; begin only on new voice"
        if self.final_task is not None:
            await self.final_task
            self.final_task = None
        self.speaking = True
        await self.writer.call(self.owner.start_utterance)
        self.utterance.clear()
        self.last_partial = 0.0
        assert self.final_task is None, "the previous utterance's last words must land first"

    def recognise_soon(self, now: float, finished: bool) -> None:
        assert self.utterance, "nothing heard yet; on_chunk collects audio before recognising"
        assert now >= self.last_partial, (
            f"clock went from {self.last_partial} back to {now}; use time.monotonic()"
        )
        self.last_partial = now
        self.serial += 1
        task = asyncio.create_task(self.recognise(np.concatenate(self.utterance), self.serial))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        if finished:
            self.final_task = task
            self.speaking = False
            self.utterance.clear()
            self.recent.clear()

    async def on_chunk(self, chunk: npt.NDArray[np.float32]) -> None:
        assert chunk.ndim == 1, f"audio chunk has shape {chunk.shape}; downmix() makes it mono"
        assert np.isfinite(chunk).all(), "audio chunk has NaN or infinite samples; check the input"
        now = time.monotonic()
        self.recent.append(math.sqrt(float(np.square(chunk).mean())))
        voice = sum(self.recent) / len(self.recent) > SILENCE_RMS
        if voice and not self.speaking:
            await self.begin_utterance()
        if not self.speaking:
            return
        self.utterance.append(chunk)
        if voice:
            self.last_voice = now
        elapsed = len(self.utterance) * FRAME_SECONDS
        finished = now - self.last_voice >= SILENCE_SECONDS or elapsed >= MAX_UTTERANCE_SECONDS
        due = finished or now - self.last_partial >= PARTIAL_INTERVAL
        if due and elapsed >= MIN_UTTERANCE_SECONDS:
            self.recognise_soon(now, finished)

    async def listen(self) -> None:
        """Feed audio to recognition until the input device stops, e.g. when it is unplugged."""
        import sounddevice as sd

        device = sd.query_devices(self.options.device, "input")
        assert device["max_input_channels"] > 0, (
            f"{device['name']} has no input channels; sounddevice should only match inputs"
        )
        assert self.frames.empty(), "audio is queued before the stream opened"
        say(f"Listening on {device['name']} at the cursor")
        with sd.InputStream(
            device=device["index"],
            samplerate=SAMPLE_RATE,
            channels=device["max_input_channels"],
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=audio_callback(self.frames),
        ) as stream:
            while stream.active:
                await asyncio.sleep(0.04)
                while not self.frames.empty():
                    await self.on_chunk(self.frames.get_nowait())
        say(f"{device['name']} stopped sending audio", "error")


def list_devices(value: bool) -> None:
    """Print input device names as one line of JSON, then exit."""
    if not value:
        return
    import sounddevice as sd

    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    out = json.dumps(names, ensure_ascii=False)
    assert "\n" not in out, "the plugin reads the device list as one line of JSON"
    assert json.loads(out) == names, "device names must survive JSON for the plugin to parse"
    print(out)
    raise typer.Exit


def serve(
    socket: Annotated[str, typer.Option(help="Neovim listen socket")],
    model: Annotated[str, typer.Option(envvar="VOICECODE_MODEL")] = "openai:gpt-4.1-mini",
    asr_model: Annotated[str, typer.Option(envvar="VOICECODE_ASR_MODEL")] = "base.en",
    device: Annotated[
        str | None,
        typer.Option(envvar="VOICECODE_DEVICE", help="Input device name, or a unique part of it"),
    ] = None,
    _list_devices: Annotated[
        bool,
        typer.Option(
            "--list-devices", callback=list_devices, is_eager=True, help="List input devices"
        ),
    ] = False,
) -> None:
    """Live ASR and code editing with one owner of Neovim buffer mutations."""
    assert socket, "no Neovim socket given; the plugin passes --socket from v:servername"
    assert model and asr_model, "model names are empty; unset VOICECODE_MODEL to use defaults"
    try:
        asyncio.run(Session.run(Options(socket, model, asr_model, device)))
    except UserError as exc:
        say(f"Cannot start model {model}: {str(exc).split('. ')[0]}", "error")
        raise typer.Exit(1) from None
    except ValueError as exc:
        say(f"Cannot open input device {device!r}: {exc}", "error")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


app = typer.Typer(add_completion=False)
_ = app.command()(serve)

if __name__ == "__main__":
    app()
