"""Live ASR and code editing with one owner of Neovim buffer mutations."""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Annotated, Protocol, final

import numpy as np
import numpy.typing as npt
import typer
from pydantic_ai.exceptions import ModelAuthenticationError, UserError
from pynvim.api import NvimError

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.1
PARTIAL_INTERVAL = 0.8
SILENCE_SECONDS = 0.9
MIN_UTTERANCE_SECONDS = 0.4
MAX_UTTERANCE_SECONDS = 20.0
SILENCE_RMS = 0.012


class Editor(Protocol):
    """Buffer text is lines joined by "\\n"; offsets are character indices into it."""

    def current(self) -> tuple[int, int]:
        """Active buffer and the insertion offset at its cursor."""
        ...

    def name(self, buffer: int) -> str: ...
    def read(self, buffer: int) -> str: ...
    def splice(self, buffer: int, span: range, text: str) -> None: ...
    def place_cursor(self, buffer: int, offset: int) -> None: ...


@dataclass
class Anchor:
    """Speech is ordinary buffer text at buffer[start:start + len(text)]."""

    buffer: int
    start: int
    text: str = ""
    skip: int = 0  # ASR words of this utterance already committed as ordinary text


@dataclass(frozen=True)
class Snapshot:
    buffer: int
    content: str
    end: int  # end of the speech region the model was shown


def map_offset(old: str, new: str, pos: int) -> int:
    """Where `pos` in `old` lands in `new`, after any change touching it."""
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal" and i1 <= pos < i2:
            return j1 + pos - i1
        if tag != "equal" and i1 <= pos <= i2:
            return j2
    return len(new)


class BufferOwner:
    """The only component allowed to mutate the Neovim buffer."""

    def __init__(self, editor: Editor) -> None:
        self.editor: Editor = editor
        self.anchor: Anchor = Anchor(-1, 0)
        self.content: str = ""
        self.mark: tuple[int, int] | None = None

    def _edited(self) -> bool:
        return self.editor.read(self.anchor.buffer) != self.content

    def _diverged(self) -> bool:
        return self.mark is None or self.editor.current() != self.mark or self._edited()

    def _anchor_at_cursor(self) -> None:
        buffer, offset = self.editor.current()
        self.anchor = Anchor(buffer, offset, skip=self.anchor.skip)
        self.content = self.editor.read(buffer)
        self.mark = (buffer, offset)

    def _write(self, span: range, text: str) -> None:
        self.editor.splice(self.anchor.buffer, span, text)
        self.content = self.content[: span.start] + text + self.content[span.stop :]

    def _follow(self) -> None:
        """Put the cursor after the speech and remember where we left it."""
        self.editor.place_cursor(self.anchor.buffer, self.anchor.start + len(self.anchor.text))
        self.mark = self.editor.current()

    def start_utterance(self) -> None:
        if self._diverged():
            self._anchor_at_cursor()
        else:
            self.anchor.start += len(self.anchor.text)
            self.anchor.text = ""
        self.anchor.skip = 0

    def partial(self, words: str) -> None:
        assert isinstance(words, str)
        a = self.anchor
        if self._diverged():
            # The user moved or edited: placed speech stays, new speech follows the cursor.
            a.skip += len(a.text.split())
            self._anchor_at_cursor()
            a = self.anchor
        text = " ".join(words.split()[a.skip :])
        old_end = a.start + len(a.text)
        if text and a.start and not self.content[a.start - 1].isspace():
            text = " " + text
        if text and old_end < len(self.content) and not self.content[old_end].isspace():
            text += " "
        a.text = text
        self._write(range(a.start, old_end), text)
        self._follow()

    def snapshot(self) -> Snapshot | None:
        a = self.anchor
        if not a.text.strip() or self._edited():
            return None
        return Snapshot(a.buffer, self.content, a.start + len(a.text))

    def finish_utterance(self) -> Snapshot | None:
        return self.snapshot()

    def apply_code(self, snap: Snapshot, replacement: str) -> bool:
        """Rebase a model edit onto speech appended since `snap`; reject anything else."""
        assert isinstance(replacement, str)
        a = self.anchor
        replacement = replacement.rstrip("\n")
        if not replacement.strip() or a.buffer != snap.buffer or self._edited():
            return False
        end = a.start + len(a.text)
        current = self.content
        if end < snap.end or current[: snap.end] != snap.content[: snap.end]:
            return False
        if current[end:] != snap.content[snap.end :]:
            return False
        moved = self.mark is None or self.editor.current() != self.mark
        new_end = map_offset(snap.content, replacement, snap.end)
        tail = current[snap.end : end]
        result = replacement[:new_end] + tail + replacement[new_end:]
        if a.start < snap.end:
            a.skip += len(current[a.start : snap.end].split())
            a.text = tail
            a.start = new_end
        else:
            a.start = new_end + a.start - snap.end
        prefix = len(os.path.commonprefix([current, result]))
        suffix = len(os.path.commonprefix([current[prefix:][::-1], result[prefix:][::-1]]))
        self._write(range(prefix, len(current) - suffix), result[prefix : len(result) - suffix])
        if not moved:
            self._follow()
        return True


def char_to_position(text: str, offset: int) -> tuple[int, int]:
    """Character offset to Neovim's zero-based (row, byte column)."""
    lines = text[:offset].split("\n")
    return len(lines) - 1, len(lines[-1].encode())


class NvimEditor:
    def __init__(self, address: str) -> None:
        import pynvim

        self.nvim: pynvim.Nvim = pynvim.attach("socket", path=address)

    def current(self) -> tuple[int, int]:
        window = self.nvim.current.window
        lines: list[str] = window.buffer[:]
        row, col = window.cursor
        line = lines[row - 1]
        chars = len(line.encode()[:col].decode(errors="ignore"))
        if not self.nvim.api.get_mode()["mode"].startswith("i") and line:
            chars = min(chars + 1, len(line))  # Normal mode inserts after the cursor, like `a`.
        return window.buffer.number, sum(len(x) + 1 for x in lines[: row - 1]) + chars

    def name(self, buffer: int) -> str:
        return str(self.nvim.buffers[buffer].name or "untitled.py")

    def read(self, buffer: int) -> str:
        return "\n".join(self.nvim.buffers[buffer][:])

    def splice(self, buffer: int, span: range, text: str) -> None:
        content = self.read(buffer)
        (r1, c1), (r2, c2) = (char_to_position(content, i) for i in (span.start, span.stop))
        self.nvim.api.buf_set_text(self.nvim.buffers[buffer], r1, c1, r2, c2, text.split("\n"))

    def place_cursor(self, buffer: int, offset: int) -> None:
        window = self.nvim.current.window
        if window.buffer.number != buffer:
            return
        content = self.read(buffer)
        normal = not self.nvim.api.get_mode()["mode"].startswith("i")
        if normal and offset and content[offset - 1] != "\n":
            offset -= 1
        row, col = char_to_position(content, offset)
        window.cursor = (row + 1, col)


@final
class Transcriber:
    def __init__(self, model: str) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="auto", compute_type="auto")

    def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        assert audio.ndim == 1
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class CodeEditor:
    def __init__(self, model: str) -> None:
        from pydantic_ai import Agent

        self.agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription. Transform the spoken instructions into "
                "working code and revise existing code as requested. Remove natural-language "
                "instructions once implemented. Preserve unrelated existing code. "
                "Return ONLY the complete replacement buffer contents, without fences, "
                "explanations, or commentary. Do not invent dependencies unless needed. "
                "If instructions are incomplete, preserve them until they can be implemented."
            ),
        )

    async def transform(self, content: str, filename: str) -> str:
        assert content.strip()
        result = await self.agent.run(
            f"Filename: {filename}\nCurrent entire buffer:\n<buffer>\n{content}\n</buffer>"
        )
        return result.output


def say(message: str, level: str = "info") -> None:
    """User-facing status for the Neovim plugin: one "level<TAB>message" line on stdout."""
    print(f"{level}\t{' '.join(message.split())}", flush=True)


def log(message: str) -> None:
    """Diagnostics that only belong in the plugin's log file."""
    print(" ".join(message.split()), file=sys.stderr, flush=True)


@final
class Writer:
    """pynvim's sync client runs its own event loop, so it cannot be called from asyncio.
    One dedicated thread performs every Neovim call, which also serialises all writes."""

    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicecode-writer")

    async def call[T](self, fn: Callable[..., T], *args: object) -> T:
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)


def downmix(indata: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Sum every input channel to mono, so aggregate devices (mic plus meeting audio) are heard."""
    return np.clip(indata.sum(axis=1, dtype=np.float32), -1.0, 1.0)


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
    transcriber: Transcriber
    coder: CodeEditor
    owner: BufferOwner = field(init=False)
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

    def __post_init__(self) -> None:
        self.owner = BufferOwner(self.editor)

    @classmethod
    async def open(cls, options: Options) -> Session:
        coder = CodeEditor(options.model)  # Fails fast on a missing API key, before the ASR load.
        writer = Writer()
        editor = await writer.call(NvimEditor, options.socket)
        say("Loading speech model")
        transcriber = await asyncio.to_thread(Transcriber, options.asr_model)
        return cls(options, writer, editor, transcriber, coder)

    def on_audio(self, indata: npt.NDArray[np.float32], *rest: object) -> None:
        """sounddevice callback: (indata, frames, time, status)."""
        if rest[-1]:
            log(f"Audio: {rest[-1]}")
        self.frames.put(downmix(indata))

    async def edit_snapshot(self) -> None:
        try:
            snapshot = await self.writer.call(self.owner.snapshot)
            if snapshot is None or snapshot.content == self.last_sent:
                return
            self.last_sent = snapshot.content
            filename = await self.writer.call(self.editor.name, snapshot.buffer)
            replacement = await self.coder.transform(snapshot.content, filename)
            if not await self.writer.call(self.owner.apply_code, snapshot, replacement):
                log("Stale edit discarded")
        except ModelAuthenticationError as exc:
            say(f"The model provider rejected the API key: {exc}", "auth")
        except (RuntimeError, NvimError) as exc:
            say(f"Code edit failed: {exc}", "error")

    def transform_soon(self) -> None:
        """Keep one model call in flight; newer speech queues exactly one rerun."""
        if self.coding is not None:
            self.rerun = True
            return
        self.coding = asyncio.create_task(self.edit_snapshot())
        self.coding.add_done_callback(self.transform_done)

    def transform_done(self, _: asyncio.Task[None]) -> None:
        self.coding = None
        if self.rerun:
            self.rerun = False
            self.transform_soon()

    async def recognise(self, audio: npt.NDArray[np.float32], sequence: int) -> None:
        """A newer ASR result supersedes an older one; only the owner edits."""
        try:
            words = await asyncio.to_thread(self.transcriber.transcribe, audio)
            if sequence != self.serial:
                return
            await self.writer.call(self.owner.partial, words)
            self.transform_soon()
        except (RuntimeError, ValueError, NvimError) as exc:
            say(f"Transcription failed: {exc}", "error")

    async def begin_utterance(self) -> None:
        if self.final_task is not None:
            await self.final_task
            self.final_task = None
        self.speaking = True
        await self.writer.call(self.owner.start_utterance)
        self.utterance.clear()
        self.last_partial = 0.0

    def recognise_soon(self, now: float, finished: bool) -> None:
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
        import sounddevice as sd

        device = sd.query_devices(self.options.device, "input")
        say(f"Listening on {device['name']} at the cursor")
        with sd.InputStream(
            device=device["index"],
            samplerate=SAMPLE_RATE,
            channels=device["max_input_channels"],
            dtype="float32",
            blocksize=1600,
            callback=self.on_audio,
        ):
            while True:
                await asyncio.sleep(0.04)
                while not self.frames.empty():
                    await self.on_chunk(self.frames.get_nowait())


async def run(options: Options) -> None:
    session = await Session.open(options)
    await session.listen()


def list_devices(value: bool) -> None:
    """Print input device names as a JSON array, then exit."""
    if not value:
        return
    import sounddevice as sd

    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    print(json.dumps(names))
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
    try:
        asyncio.run(run(Options(socket, model, asr_model, device)))
    except UserError as exc:
        say(f"Cannot start model {model}: {str(exc).split('. ')[0]}", "error")
        raise typer.Exit(1) from None
    except ValueError as exc:
        say(f"Cannot open input device {device!r}: {exc}", "error")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
