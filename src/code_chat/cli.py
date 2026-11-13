"""Search code by describing it out loud, from any terminal or editor.

    code-chat [ROOT]           listen, show matches live, open the chosen one in
                               $VISUAL / $EDITOR at its line, or print path:line
                               when neither is set or stdout isn't a terminal
    code-chat --print [ROOT]   ... always print path:line
    code-chat --open [ROOT]    ... always open it, in vi if no editor is set
    code-chat --list [ROOT]    listen, then print every match as path:line:col: text
    code-chat serve [ROOT]     the JSON-lines engine, over stdin and stdout, for an editor
    code-chat connect [ROOT]   the same, through the daemon, sharing its loaded model
    code-chat daemon           the same engine over a socket, keeping the model loaded
    code-chat stop             stop the daemon

The live view draws on the terminal (stderr), so stdout carries only the
result: `nvim $(code-chat)`, `code -g $(code-chat)`, `vim -q <(code-chat --list)`.
Searches go through the daemon, which the first one starts, so only that one
waits for the model to load.
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from typing import TextIO, cast

# Keep model-loading chatter out of the daemon's log.
_ = os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_ = os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_ = os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")


from code_chat import daemon, server
from code_chat.index import Hit
from code_chat.view import Search, shown

Message = dict[str, object]

# Editors that take file:line as one argument; the rest take +line file.
COLON_EDITORS = ("zed", "subl", "hx", "helix", "micro")
GOTO_EDITORS = ("code", "code-insiders", "cursor", "windsurf", "codium")
COMMANDS = ("serve", "connect", "daemon", "stop")
DESCRIPTION = "Search code by describing it out loud, from any terminal or editor."


def main():
    assert sys.argv, "main() needs the command line in sys.argv; run it as the code-chat script"
    command = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in COMMANDS else None
    if command:
        return run_command(command, sys.argv[2:])
    parser = argparse.ArgumentParser(prog="code-chat", description=DESCRIPTION)
    _ = parser.add_argument(
        "root", nargs="?", help="repository to search (default: the git repo you're in)"
    )
    output = parser.add_mutually_exclusive_group()
    _ = output.add_argument(
        "--open",
        action="store_true",
        help="open the chosen match in $VISUAL / $EDITOR, even when stdout isn't a terminal",
    )
    _ = output.add_argument(
        "--print",
        action="store_true",
        help="print the chosen match as path:line, even when an editor is set",
    )
    _ = output.add_argument(
        "--list", action="store_true", help="print every match, grep style, without choosing"
    )
    args = parser.parse_args()
    out = result_stream()
    given = cast(str | None, args.root)
    root = Path(given).resolve() if given else repository(Path.cwd().resolve())
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    listing = cast(bool, args.list)
    search = Search(root, choose=not listing)
    hits = search.run()
    if not hits:
        # The live view is gone by now, so say why there's nothing.
        if search.reason:
            print(f"code-chat: {search.reason}", file=sys.stderr)
        sys.exit(1)
    if listing:
        print("\n".join(grep_line(hit) for hit in hits), file=out)
    elif editor := chosen_editor(
        out, opening=cast(bool, args.open), printing=cast(bool, args.print)
    ):
        out.close()
        command = editor_command(editor, hits[0]["file"], hits[0]["start"])
        os.execvp(command[0], command)
    else:
        print(f"{shown(hits[0]['file'])}:{hits[0]['start']}", file=out)


def chosen_editor(out: TextIO, opening: bool, printing: bool) -> str | None:
    """The editor to open the match in, or None to print it. An editor set in
    $VISUAL or $EDITOR opens it, unless stdout is captured, as in
    `nvim $(code-chat)`, which wants the path."""
    assert not (opening and printing), (
        f"got --open={opening} and --print={printing}; pass at most one, as argparse enforces"
    )
    assert not out.closed, (
        "chosen_editor() checks whether the result stream is a terminal; call it before closing that"
    )
    editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR") or "").strip()
    if opening:
        return editor or "vi"
    if printing or not editor or not out.isatty():
        return None
    return editor


def run_command(command: str, rest: list[str]):
    """code-chat serve, connect, daemon or stop."""
    assert command in COMMANDS, f"{command!r} isn't a command; use one of {COMMANDS}"
    assert len(rest) <= 1, f"{command} takes at most a repository, got {rest}; pass one path"
    if command == "serve":
        server.main(Path(rest[0] if rest else ".").resolve())
    elif command == "connect":
        bridge(Path(rest[0] if rest else ".").resolve())
    elif command == "daemon":
        daemon.serve_forever()
    else:
        print("stopped" if daemon.stop() else "no daemon was running", file=sys.stderr)


def bridge(root: Path):
    """Relay the protocol between stdin and stdout and the daemon, so an
    editor can talk to the daemon as it would to `code-chat serve`. Closing
    stdin hangs up, which frees the daemon for the next search."""
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    out = server.protect_stdout()
    connection, messages = daemon.connect(root)
    with connection:
        threading.Thread(target=forward, args=(sys.stdin, connection), daemon=True).start()
        try:
            for line in messages:
                _ = out.write(line)
                out.flush()
        except BrokenPipeError:
            # The editor went away; hanging up below ends the search.
            return
    assert connection.fileno() == -1, (
        "the daemon connection should be closed by now; keep the relay inside the with block"
    )


def forward(lines: TextIO, connection: socket.socket):
    """Send the editor's commands on to the daemon, then hang up our side."""
    assert connection.family == socket.AF_UNIX, (
        f"forwarding to a {connection.family} socket; connect to the daemon's Unix socket"
    )
    assert lines.readable(), f"can't read commands from {lines}; pass stdin"
    try:
        for line in lines:
            connection.sendall(line.encode())
        connection.shutdown(socket.SHUT_WR)
    except OSError:
        # The daemon hung up first; the relay ends with it.
        return


def result_stream() -> TextIO:
    """Only the result goes to stdout; anything else written there, by Python
    libraries or C ones like PortAudio, goes to stderr, so `$(code-chat)`
    gets just the path."""
    assert sys.stdout is not sys.stderr, "stdout is already redirected; call result_stream() once"
    _ = sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    _ = os.dup2(2, 1)
    assert out.fileno() > 2, (
        f"the result stream got descriptor {out.fileno()}, one of the standard three; call result_stream() once"
    )
    return out


def repository(cwd: Path) -> Path:
    """The git repository `cwd` is in, or `cwd` itself outside one."""
    assert cwd.is_absolute(), f"working directory {cwd} must be absolute; resolve() it first"
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        root = Path(top.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        root = cwd
    assert cwd.is_relative_to(root), (
        f"repository {root} should contain the working directory {cwd}; pass the repository explicitly"
    )
    return root


def grep_line(hit: Hit) -> str:
    """path:line:col: text, as grep -n and vim's errorformat read it."""
    assert hit["start"] >= 1, (
        f"hit in {hit['file']} starts at line {hit['start']}; send 1-based lines"
    )
    symbols = ", ".join(hit["symbols"])
    out = f"{shown(hit['file'])}:{hit['start']}:1: {symbols + ': ' if symbols else ''}{hit['preview']}"
    assert "\n" not in out, (
        f"grep output for {hit['file']} must be one line, got {out!r}; strip newlines from the preview"
    )
    return out


def editor_command(editor: str, file: str, line: int) -> list[str]:
    """The command that opens `file` at `line` in `editor`."""
    assert editor.strip(), f"no editor to open with (got {editor!r}); set $VISUAL or $EDITOR"
    assert line >= 1, f"line {line} in {file} must count from 1; pass Hit['start']"
    command = editor.split()
    name = Path(command[0]).name
    if name in GOTO_EDITORS:
        return command + ["--goto", f"{file}:{line}"]
    if name in COLON_EDITORS:
        return command + [f"{file}:{line}"]
    # vi, vim, nvim, emacs, nano, kak and most others.
    return command + [f"+{line}", file]


# --- Input -------------------------------------------------------------------------


if __name__ == "__main__":
    main()
