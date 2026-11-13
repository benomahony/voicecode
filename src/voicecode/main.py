"""Live ASR and code editing with one owner of Neovim buffer mutations."""

from __future__ import annotations

import asyncio
import math
import os
import queue
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Annotated, Protocol

import numpy as np
import numpy.typing as npt
import typer

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
    def splice(self, buffer: int, start: int, end: int, text: str) -> None: ...
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

    def _write(self, start: int, end: int, text: str, cursor: int, move: bool = True) -> None:
        self.editor.splice(self.anchor.buffer, start, end, text)
        self.content = self.content[:start] + text + self.content[end:]
        if move:
            self.editor.place_cursor(self.anchor.buffer, cursor)
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
        self._write(a.start, old_end, text, a.start + len(text))

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
        self._write(
            prefix,
            len(current) - suffix,
            result[prefix : len(result) - suffix],
            a.start + len(a.text),
            move=not moved,
        )
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

    def splice(self, buffer: int, start: int, end: int, text: str) -> None:
        content = self.read(buffer)
        (r1, c1), (r2, c2) = char_to_position(content, start), char_to_position(content, end)
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


class Transcriber:
    def __init__(self, model: str) -> None:
        from faster_whisper import WhisperModel

        self.model: WhisperModel = WhisperModel(model, device="auto", compute_type="auto")

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


async def run(address: str, model: str, asr_model: str) -> None:
    import sounddevice as sd
    from pynvim.api import NvimError

    # pynvim's sync client runs its own event loop, so it cannot be called from this one.
    # One dedicated thread performs every Neovim call, which also serialises all writes.
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicecode-writer")
    loop = asyncio.get_running_loop()

    async def on_writer[T](fn: Callable[..., T], *args: object) -> T:
        return await loop.run_in_executor(writer, fn, *args)

    editor = await on_writer(NvimEditor, address)
    owner = BufferOwner(editor)
    transcriber = await asyncio.to_thread(Transcriber, asr_model)
    coder = CodeEditor(model)
    frames: queue.SimpleQueue[npt.NDArray[np.float32]] = queue.SimpleQueue()
    tasks: set[asyncio.Task[None]] = set()
    coding: asyncio.Task[None] | None = None
    rerun = False
    last_sent = ""

    def on_audio(
        indata: npt.NDArray[np.float32], frames_count: int, _time: object, status: object
    ) -> None:
        assert frames_count > 0
        if status:
            print(f"Audio: {status}", file=sys.stderr)
        frames.put(indata[:, 0].copy())

    async def edit_snapshot() -> None:
        nonlocal last_sent
        try:
            snapshot = await on_writer(owner.snapshot)
            if snapshot is None or snapshot.content == last_sent:
                return
            last_sent = snapshot.content
            filename = await on_writer(editor.name, snapshot.buffer)
            replacement = await coder.transform(snapshot.content, filename)
            if not await on_writer(owner.apply_code, snapshot, replacement):
                print("Stale edit discarded", file=sys.stderr)
        except (RuntimeError, NvimError) as exc:
            print(f"Code transformation failed: {exc}", file=sys.stderr)

    def transform_soon() -> None:
        """Keep one model call in flight; newer speech queues exactly one rerun."""
        nonlocal coding, rerun
        if coding is not None:
            rerun = True
            return

        def done(_: asyncio.Task[None]) -> None:
            nonlocal coding, rerun
            coding = None
            if rerun:
                rerun = False
                transform_soon()

        coding = asyncio.create_task(edit_snapshot())
        coding.add_done_callback(done)

    utterance: list[npt.NDArray[np.float32]] = []
    speaking = False
    last_voice = 0.0
    last_partial = 0.0
    serial = 0
    final_task: asyncio.Task[None] | None = None
    recent: deque[float] = deque(maxlen=3)

    async def recognise(audio: npt.NDArray[np.float32], sequence: int) -> None:
        """A newer ASR result supersedes an older one; only the owner edits."""
        try:
            words = await asyncio.to_thread(transcriber.transcribe, audio)
            if sequence != serial:
                return
            await on_writer(owner.partial, words)
            transform_soon()
        except (RuntimeError, ValueError, NvimError) as exc:
            print(f"Transcription failed: {exc}", file=sys.stderr)

    print("Voicecode listening at the cursor. Ctrl-C to stop.", file=sys.stderr)
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=1600, callback=on_audio
    ):
        while True:
            await asyncio.sleep(0.04)
            while not frames.empty():
                chunk = frames.get_nowait()
                now = time.monotonic()
                rms = math.sqrt(float(np.square(chunk).mean()))
                recent.append(rms)
                voice = sum(recent) / len(recent) > SILENCE_RMS
                if voice and not speaking:
                    if final_task is not None:
                        await final_task
                        final_task = None
                    speaking = True
                    await on_writer(owner.start_utterance)
                    utterance.clear()
                    last_partial = 0.0
                if not speaking:
                    continue
                utterance.append(chunk)
                if voice:
                    last_voice = now
                elapsed = len(utterance) * FRAME_SECONDS
                finished = (now - last_voice >= SILENCE_SECONDS) or (
                    elapsed >= MAX_UTTERANCE_SECONDS
                )
                if not finished and now - last_partial < PARTIAL_INTERVAL:
                    continue
                if elapsed < MIN_UTTERANCE_SECONDS:
                    continue
                last_partial = now
                serial += 1
                current = serial
                audio = np.concatenate(utterance)
                task = asyncio.create_task(recognise(audio, current))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                if finished:
                    final_task = task
                    speaking = False
                    utterance.clear()
                    recent.clear()


def serve(
    socket: Annotated[str, typer.Option(help="Neovim listen socket")],
    model: Annotated[str, typer.Option(envvar="VOICECODE_MODEL")] = "openai:gpt-4.1-mini",
    asr_model: Annotated[str, typer.Option(envvar="VOICECODE_ASR_MODEL")] = "base.en",
) -> None:
    """Live ASR and code editing with one owner of Neovim buffer mutations."""
    try:
        asyncio.run(run(socket, model, asr_model))
    except KeyboardInterrupt:
        pass


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
