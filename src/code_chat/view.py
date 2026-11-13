"""The terminal live view: matches as syntax-highlighted cards beside a waveform,
drawn on stderr while the daemon ranks what you say."""

import json
import math
import os
import queue
import select
import socket
import termios
import threading
import time
import tty
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, cast, final

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from code_chat import daemon
from code_chat.index import Hit

Message = dict[str, object]

CARD_LINES = 9
BLOCKS = "▁▂▃▄▅▆▇█"
QUIT = ("\x1b", "q", "\x03")
DOWN = ("j", "\x1b[B", "\x0e", "\t")
UP = ("k", "\x1b[A", "\x10", "\x1b[Z")


def shown(file: str) -> str:
    """Relative to where you are when it's inside, so output reads well."""
    path = Path(file)
    assert path.is_absolute(), (
        f"hit paths come from the daemon as absolute paths, got {file!r}; send Hit['file']"
    )
    out = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else file
    assert out.endswith(path.name), (
        f"shortening {file} to {out} lost the file name; keep the last path part"
    )
    return out


@contextmanager
def keys() -> Generator["queue.Queue[str]"]:
    """Keypresses from the terminal, without echo or waiting for Enter."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        # No terminal, as in a pipeline: no keys, but listening still works.
        yield queue.Queue()
        return
    assert fd >= 0, f"opening /dev/tty gave descriptor {fd}; run code-chat from a terminal"
    saved = termios.tcgetattr(fd)
    _ = tty.setcbreak(fd)
    pressed: queue.Queue[str] = queue.Queue()
    # Writing to this pipe tells the reader to stop, in the same select() it
    # waits for keys in, so a stop can't slip between checking and reading.
    stop_reading, stop_writing = os.pipe()
    reader = threading.Thread(target=read_keys, args=(fd, stop_reading, pressed), daemon=True)
    reader.start()
    try:
        yield pressed
    finally:
        _ = os.write(stop_writing, b"x")
        reader.join()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        for open_fd in (fd, stop_reading, stop_writing):
            os.close(open_fd)
    assert not reader.is_alive(), (
        "the key reader must stop before the terminal is restored; signal it through the pipe"
    )


def read_keys(fd: int, stop: int, pressed: "queue.Queue[str]"):
    """Queue each keypress (or escape sequence) until `stop` is readable."""
    assert fd >= 0 and stop >= 0, (
        f"read_keys() needs open descriptors, got terminal {fd} and stop pipe {stop}"
    )
    assert fd != stop, (
        f"the terminal and the stop pipe must be different descriptors, both are {fd}"
    )
    ready: list[int] = []
    while stop not in ready:
        ready = select.select([fd, stop], [], [])[0]
        if fd in ready:
            pressed.put(os.read(fd, 8).decode(errors="ignore"))


def read_messages(stream: TextIO, messages: "queue.Queue[Message]"):
    """Queue each message from the daemon, then a hang-up when it closes."""
    assert stream.readable(), (
        f"read_messages() needs the daemon's message stream, got {stream}; use daemon.connect()"
    )
    assert messages.empty(), (
        "read_messages() must start before any message is queued; start it once per connection"
    )
    try:
        for line in stream:
            messages.put(cast(Message, json.loads(line)))
    except (OSError, ValueError) as error:
        messages.put({"type": "hangup", "error": str(error)})
        return
    messages.put({"type": "hangup", "error": f"the daemon closed the connection; see {daemon.LOG}"})


def drain[T](items: "queue.Queue[T]") -> list[T]:
    """Everything queued since the last call."""
    assert items.maxsize == 0, (
        f"drain() expects an unbounded queue, got maxsize {items.maxsize}; make it with queue.Queue()"
    )
    out: list[T] = []
    while not items.empty():
        out.append(items.get())
    nones = sum(item is None for item in out)
    assert nones == 0, f"{nones} of {len(out)} queued items are None; put a value or nothing"
    return out


# --- The search ----------------------------------------------------------------------


@final
class Search:
    def __init__(self, root: Path, choose: bool):
        assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
        assert root.is_dir(), (
            f"repository root {root} isn't a directory; pass the repository to search"
        )
        self.root = root
        self.choose = choose
        self.console = Console(stderr=True)
        self.status = "Connecting to the code-chat daemon…"
        self.ready = False
        self.listening = False
        self.level = 0.0
        self.seconds = 0.0
        self.hits: list[Hit] = []
        self.selected = 0
        self.heights = [0.0] * 24
        self.connection: socket.socket | None = None
        # Set when the search ends: the chosen hits, or [] when cancelled,
        # with the reason when it ended without any.
        self.outcome: list[Hit] | None = None
        self.reason: str | None = None

    def run(self) -> list[Hit]:
        assert self.outcome is None, "a Search runs once; make a new one to search again"
        assert self.connection is None, "run() opens its own connection; don't connect first"
        messages: queue.Queue[Message] = queue.Queue()
        with (
            keys() as pressed,
            Live(self.view(), console=self.console, transient=True, refresh_per_second=25) as live,
        ):
            live.update(self.view())
            connection, stream = daemon.connect(self.root)
            self.connection = connection
            threading.Thread(target=read_messages, args=(stream, messages), daemon=True).start()
            while self.outcome is None:
                self.handle(drain(pressed), drain(messages))
                live.update(self.view())
                time.sleep(0.03)
        # Hanging up stops the daemon listening, and frees it for the next search.
        connection.close()
        return self.outcome

    def handle(self, pressed: list[str], messages: list[Message]):
        """Act on keys, then messages, until one of them ends the search: a
        batch can hold more after that, like the final results followed by
        listening stopping."""
        assert self.outcome is None, (
            "handle() is for a search that's still going; stop the loop when it ends"
        )
        assert all(pressed), (
            f"got empty keypresses {pressed}; read_keys() should only queue what it read"
        )
        for key in pressed:
            self.press(key)
            if self.outcome is not None:
                return
        for message in messages:
            self.receive(message)
            if self.outcome is not None:
                return

    def send(self, message: Message):
        assert self.connection is not None, "send() needs the connection run() opens"
        assert message.get("type"), f"message {message} has no type; the daemon dispatches on it"
        try:
            self.connection.sendall((json.dumps(message) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError) as error:
            self.status = f"Lost the daemon ({error}); see {daemon.LOG}"

    def press(self, key: str):
        assert key, "press() needs a key; skip empty reads"
        assert self.outcome is None, (
            f"key {key!r} arrived after the search ended; stop reading keys when it does"
        )
        if key in QUIT:
            self.outcome = []
            self.reason = None
        elif key in ("\r", "\n") and self.hits:
            self.outcome = [self.hits[self.selected]]
        elif key in DOWN and self.hits:
            self.selected = min(self.selected + 1, len(self.hits) - 1)
        elif key in UP:
            self.selected = max(self.selected - 1, 0)
        elif key == " " and self.ready and not self.listening:
            self.send({"type": "listen"})

    def receive(self, message: Message):
        """Act on one message from the daemon."""
        kind = message.get("type")
        assert isinstance(kind, str), f"message {message} has no type; the daemon must send one"
        assert self.outcome is None, (
            f"message {kind!r} arrived after the search ended; stop reading when it does"
        )
        if kind == "status":
            self.status = str(message["text"])
        elif kind == "ready":
            self.ready = True
            self.send({"type": "listen"})
        elif kind == "listening":
            self.listening = bool(message["on"])
            self.level = 0.0
            self.status = (
                "Listening… describe the code  (esc cancels)" if self.listening else self.status
            )
            # Listing has no one to press space and try again.
            if not self.listening and not self.choose and self.outcome is None:
                self.end(self.status)
        elif kind == "level":
            self.level = float(cast(float, message["value"]))
        elif kind == "results":
            self.show(
                cast(list[Hit], message["hits"]),
                float(cast(float, message["seconds"])),
                bool(message["final"]),
            )
        elif kind == "hangup":
            self.end(str(message["error"]))

    def end(self, reason: str):
        """End the search without a result, saying why."""
        assert reason, "end() needs a reason to tell the user; pass the status or error"
        assert self.outcome is None, (
            f"the search already ended; can't end it again because {reason!r}"
        )
        self.status = self.reason = reason
        self.outcome = []

    def show(self, hits: list[Hit], seconds: float, final: bool):
        """New rankings, keeping a match you've moved to selected."""
        backwards = [
            (hit["path"], hit["start"], hit["end"]) for hit in hits if hit["start"] > hit["end"]
        ]
        assert not backwards, f"hits end before they start: {backwards[:3]}; check Index.hit()"
        assert seconds >= 0, (
            f"heard {seconds} s of speech; the daemon reports a length, which can't be negative"
        )
        current = self.hits[self.selected] if self.hits and self.selected else None
        self.hits, self.seconds = hits, seconds
        self.selected = next(
            (
                i
                for i, h in enumerate(hits)
                if current and (h["file"], h["start"]) == (current["file"], current["start"])
            ),
            0,
        )
        if not final:
            self.status = f"Listening… {seconds:.1f} s"
        elif self.choose:
            self.status = "Closest to what you said  (↑/↓ choose · enter · space speak again · esc)"
        else:
            self.outcome = hits

    # --- Drawing -------------------------------------------------------------

    def waveform(self) -> Text:
        """Bars driven by the microphone level, tallest in the middle and each
        wobbling at its own pace, so it reads as a voice."""
        t = time.monotonic()
        level = self.level if self.listening else 0.0
        assert 0 <= level <= 1, (
            f"loudness {level} must be between 0 and 1; the daemon sends Session.loudness()"
        )
        n = len(self.heights)
        bars: list[str] = []
        for i in range(n):
            x = (i - (n - 1) / 2) / (n * 0.3)
            wobble = 0.55 + 0.45 * math.sin(t * (5 + (i * 7) % 5) + i * 2.399)
            target = level * (0.15 + 0.85 * math.exp(-x * x)) * wobble
            h = self.heights[i]
            # Rise quickly, fall back gently.
            self.heights[i] = h + (target - h) * (0.5 if target > h else 0.15)
            bars.append(BLOCKS[min(len(BLOCKS) - 1, int(self.heights[i] * len(BLOCKS)))])
        assert len(bars) == n, f"drew {len(bars)} bars for a {n}-bar waveform; draw one per height"
        return Text("󰍬 " + "".join(bars) + "  ", style="green" if level else "dim")

    def view(self) -> RenderableType:
        count_hits = len(self.hits)
        assert 0 <= self.selected < max(1, count_hits), (
            f"selection {self.selected} is outside {count_hits} hits; clamp it in press()"
        )
        header = Panel(
            self.waveform() + Text(self.status, style="dim"),
            border_style="dim",
            title="code chat",
            title_align="left",
        )
        if not self.hits:
            return header
        rows = self.console.size.height - 4
        count = max(1, min(count_hits, rows // (CARD_LINES + 2)))
        first = min(max(0, self.selected - count + 1), count_hits - count)
        assert first <= self.selected < first + count, (
            f"cards {first}-{first + count} must include selection {self.selected}; fix the scroll arithmetic"
        )
        return Group(header, *[self.card(i) for i in range(first, first + count)])

    def card(self, i: int) -> Panel:
        assert 0 <= i < len(self.hits), (
            f"card {i} is outside {len(self.hits)} hits; only draw cards for hits"
        )
        hit = self.hits[i]
        file, start, end = hit["file"], hit["start"], hit["end"]
        assert start <= end, (
            f"hit in {file} ends at {end}, before it starts at {start}; check Index.hit()"
        )
        try:
            lines = Path(file).read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            lines = [f"(can't read {file} any more)"]
        code = "\n".join(lines[start - 1 : start - 1 + CARD_LINES])
        selected = i == self.selected
        syntax = Syntax(
            code,
            Syntax.guess_lexer(file, code),
            line_numbers=True,
            start_line=start,
            highlight_lines=set(range(start, end + 1)) if selected else None,
            theme="ansi_dark",
            background_color="default",
            word_wrap=False,
        )
        symbols = f"  [bold]{', '.join(hit['symbols'])}[/]" if hit["symbols"] else ""
        best, worst = self.hits[0]["score"], self.hits[-1]["score"]
        filled = round(10 * ((hit["score"] - worst) / (best - worst) if best > worst else 1))
        return Panel(
            syntax,
            title=f"[{'reverse ' if selected else ''}bold] #{i + 1} [/] {shown(file)}:{start}{symbols}",
            title_align="left",
            subtitle=f"[green]{'▰' * filled}[/][dim]{'▱' * (10 - filled)} {hit['score']:.3f}[/]",
            subtitle_align="right",
            border_style="cyan" if selected else "dim",
            height=CARD_LINES + 2,
        )
