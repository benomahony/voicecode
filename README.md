# voicecode

Talk to Neovim, two ways:

- **Dictate code.** Speech is transcribed live **into the buffer** at the cursor, and a
  coding model turns the spoken instructions into code while you are still talking.
  One Python process is the sole writer for speech and AI edits.
- **Search code out loud** (code-chat). Describe the code and the best-matching chunks
  of the repository appear live as you speak. There's no transcription: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
  embeds your voice with its audio encoder and compares it directly against the code.

## Install

Requires Python 3.12+, uv, Neovim, microphone permission, and an API key for the coding
model. The first run downloads the `base.en` Whisper model, and the first search\nEmbeddingGemma 2; search is fastest on Apple silicon (MPS). On macOS, PortAudio
may need installing: `brew install portaudio`.

With [lazy.nvim](https://github.com/folke/lazy.nvim):

```lua
{ "benomahony/voicecode" },
```

Or with the built-in package manager (Neovim 0.12+):

```lua
vim.pack.add({ "https://github.com/benomahony/voicecode" })
```

No `setup()` call or build step is needed. The plugin launches the backend with
`uv run --project <plugin dir> voicecode`, which installs the Python dependencies
from `uv.lock` on first start.

### Keymaps

| Key / command | Action |
| --- | --- |
| `<leader>vc` | Toggle listening |
| `:VoicecodeToggle` | Toggle listening |
| `:VoicecodeStart` / `:VoicecodeStop` | Start or stop listening |
| `<Plug>(voicecode-toggle)` | Toggle, for your own mapping |
| `:VoicecodeLog` | Open the backend log |
| `:VoicecodeDevice [name]` | Choose the input device (picker without a name) |

The default keymap is only created when the key is unmapped. Change or disable it
before the plugin loads:

```lua
vim.g.voicecode_keymap = "<leader>v" -- or false to disable
```

Notifications are limited to start, stop and errors. Everything else the backend
prints, including dependency installation and model downloads, goes to
`stdpath("log")/voicecode.log`, which `:VoicecodeLog` opens.

To override the backend command, set
`vim.g.voicecode_cmd = { "/path/to/voicecode" }`.

### Input device

Voicecode listens to the system default input unless told otherwise. Pick another
with `:VoicecodeDevice` (a picker, or `:VoicecodeDevice Shure` with completion); a
running session restarts on the new device. To set it up front, use
`vim.g.voicecode_device = "Shure"` or `VOICECODE_DEVICE`. Any unique part of the
name works. `voicecode --list-devices` prints what is available.

Every channel of the device is mixed to mono, so multi-channel devices work.

#### Pairing and mobbing

To transcribe remote participants as well as the room, route the meeting audio
into an input and combine it with your microphone (macOS):

1. Install a loopback driver: `brew install blackhole-2ch`.
2. In Audio MIDI Setup, create a **Multi-Output Device** with your speakers or
   headphones and BlackHole, and select it as the meeting app's speaker, so you
   still hear everyone.
3. Create an **Aggregate Device** with your microphone and BlackHole (enable drift
   correction on BlackHole).
4. `:VoicecodeDevice Aggregate`.

Everything said by anyone becomes buffer text, so agree who is driving.

### API key

The default model is `openai:gpt-4.1-mini`. Its key is taken from, in order:

1. `OPENAI_API_KEY` in the environment Neovim was started from.
2. `vim.g.voicecode_api_key_cmd`, a command that prints the key, for example
   `{ "security", "find-generic-password", "-s", "openai", "-w" }` or
   `{ "op", "read", "op://Private/OpenAI/credential" }`.
3. A hidden prompt on start.

The key is kept for the current Neovim session only and passed to the backend's
environment; it is never written to disk. If the provider rejects it, the next
start asks again.

`VOICECODE_MODEL` can select another Pydantic AI model, which reads its own
provider's environment variable (for example `ANTHROPIC_API_KEY`). `VOICECODE_ASR_MODEL`
selects the local faster-whisper model (default `base.en`).

Open a Python file, then press `<leader>vc` or run `:VoicecodeStart`. Speak naturally. Your words appear
as ordinary text at the cursor of the active buffer, corrected as ASR stabilises.
In normal mode speech goes after the character under the cursor, like `a`. Move the
cursor or switch buffers and the next words follow it; words already placed stay put.
While you are still speaking, the coding model repeatedly receives the buffer and
replaces the spoken instruction with code. A result is applied only if the buffer has
changed solely by speech appended since it was requested; that speech is kept after
the generated code. Any other change, including a revised transcription or a manual
edit, discards the result. Press `<leader>vc` again or run `:VoicecodeStop` to stop the microphone and process.

## Limitations of this first vertical slice

- The recognizer starts after speech is detected using an RMS threshold, so quiet
  speech may require changing `SILENCE_RMS` in `main.py`.
- The agent can modify any part of the existing buffer.
- Once a code edit consumes speech, later partials drop that many leading ASR words.
  If the recognizer re-segments earlier words, a word may be dropped or repeated.
- ASR partials are re-transcribed from the beginning of the current utterance.
  This is functional but not efficient for long uninterrupted speech.
- One model call is in flight at a time; speech arriving meanwhile queues one rerun.
- The coding model returns a full replacement buffer. It is rebased only over
  appended speech; any other intervening change discards it. This favours safety
  over latency.
- Model output is not automatically typechecked or executed. Review generated code.
- Microphone and Neovim integration require a real local environment; unit tests
  cover buffer ownership, cursor placement and stale edit rejection.

## Searching code by voice (code-chat)

### Neovim

Needs Neovim 0.11+. While you speak, the best-matching chunks appear as a stack of live, syntax-highlighted code cards beside a waveform, and the window you came from follows the top match, with it highlighted.

The plugin installed above also provides `:CodeChat`:

```lua
vim.keymap.set("n", "<leader>sv", "<cmd>CodeChat<cr>", { desc = "[S]earch code by [V]oice" })
```

`:CodeChat` opens the cards and starts listening. Describe the code, e.g. "where we retry the download when YouTube refuses". Listening stops after a 4 s pause, or after 12 s if you don't say anything.

| Key | Does |
| --- | --- |
| `j` / `k` (or `<C-n>` / `<C-p>`, `<Tab>`) | Move between matches; the main window follows |
| `1`–`9` | Pick a visible card |
| `<CR>` | Go to the match, with every match in the quickfix list (`]q` / `[q` step through them) |
| `<Space>` | Speak again |
| `<Esc>` / `q` | Close and go back to where you were |

Matches stay highlighted after `<CR>` until `:CodeChatClear`. `<C-o>` returns to where you started.

When Neovim opens in a git repository, or `:cd`s into one, code-chat indexes it in the background, so the first search doesn't wait. Only what changed since last time is embedded. Neovim shares the terminal's background daemon (see below), so once one of them has loaded the model, a new Neovim is ready in well under a second.

`vim.pack` installs a local path by cloning it, so changes here reach Neovim once they're committed and you run `vim.pack.update()`.

`require("code-chat").setup({ cards = 4, width = 0.5, context = 3, index_on_open = true })` changes how many cards show at once, the share of the screen they take, how many lines show above a match in the main window, and whether to index on opening a project.

### Terminal

```sh
uv tool install --editable /path/to/voicecode
```

`code-chat` listens in the repository you're in, showing the matches live as syntax-highlighted cards beside a waveform. When you stop talking, `↑`/`↓` (or `j`/`k`) choose one, `<Enter>` opens it, `<Space>` speaks again and `<Esc>` quits.

`<Enter>` opens the match at its line in `$VISUAL` or `$EDITOR`, e.g. `nvim +12 file` or `code --goto file:12` (Cursor, Windsurf, Zed, Sublime, Helix and others get their own syntax). With neither set, or when stdout is captured as in `$(code-chat)`, it prints `path:line` instead.

The live view draws on stderr, so stdout carries only the result:

```sh
nvim $(code-chat)              # or vim, hx, zed: open the match you chose
code -g $(code-chat)           # VS Code, Cursor and the like
code-chat --print              # print path:line even with an editor set
code-chat --open               # open it even when stdout isn't a terminal (vi if no editor)
code-chat --list               # every match, as path:line:col: text
vim -q <(code-chat --list)     # ...into Vim's quickfix list
code-chat ~/Code/some/repo     # search another repository
```

It exits 1, printing nothing on stdout, when you cancel or nothing was heard, and says why on stderr.

The model stays loaded between searches in a background daemon, which the first search starts: that one waits several seconds for EmbeddingGemma 2 to load, and later ones start listening straight away. The daemon serves one search at a time over a socket in `~/.cache/code-chat/`, logs to `daemon.log` there, and exits after 30 minutes without a search. It restarts itself when code-chat's code changes, so an editable install never runs stale code. `code-chat stop` stops it sooner.

### Any editor

`--list` prints matches in the `path:line:col: text` format grep and compilers use, which most editors can jump through: Vim's quickfix list, Emacs' `grep-mode` and `compilation-mode`, and the clickable paths in VS Code's terminal.

For a live integration like the Neovim plugin, run `code-chat serve [ROOT]` and talk to it in JSON, one message per line, over stdin and stdout. `code-chat connect [ROOT]` speaks the same over stdin and stdout but through the shared daemon, which is what Neovim uses; hang up (close stdin) when the search ends, as the daemon serves one connection at a time. Or connect to the daemon's socket, `~/.cache/code-chat/daemon.sock`, to share its loaded model: open with `{"type": "open", "root": "/abs/repo", "fingerprint": "..."}`, wait for `{"type": "opened"}` (a `{"type": "restarting"}` means the daemon is older than your code; connect again), then speak the same protocol. `src/code_chat/daemon.py` has the client side.

| Direction | Message |
| --- | --- |
| In | `{"type": "listen"}`, `{"type": "stop"}` |
| In | `{"type": "update", "file": "/abs/path"}` after a save, to re-index that file |
| In | `{"type": "exit"}`, or close stdin |
| Out | `{"type": "status", "text": "Loading EmbeddingGemma 2…"}` while starting |
| Out | `{"type": "ready", "chunks": 1234}` |
| Out | `{"type": "listening", "on": true}` and, about 20 times a second, `{"type": "level", "value": 0.7}` |
| Out | `{"type": "results", "final": false, "seconds": 2.2, "hits": [...]}` on each re-rank; `final` once listening ends |

Each hit has `file` (absolute), `path` (relative to the repository), `start`, `end` and `end_col` (1-based lines, inclusive), `symbols`, a one-line `preview` and its `score`. `lua/code-chat/init.lua` is a complete client.

### How it works

- **Index.** Files come from `git ls-files` (tracked and untracked, minus ignored). [tree-sitter](https://github.com/kreuzberg-dev/tree-sitter-language-pack) splits each into chunks of about a function or two, with 40-line windows for files it can't parse. Each chunk is embedded as `title: <path> | text: <code>`. Vectors are cached by content in `~/.cache/code-chat/`, and a file is re-embedded when the editor reports a save.
- **Listening.** The microphone streams 16 kHz mono. Speech is whatever stands well clear of the room's background level. Once you've said 1.5 s, what you've said so far is embedded again after each further ¾ s, and chunks rank by cosine to it. That's averaged with earlier updates, so results only move when they're consistently better.

## Development

```sh
uv run pytest -q
uv run basedpyright
zanity check .
```

`zanity.toml` selects every zanity rule, as `zanity check --production` does.
`typings/` holds local type stubs for packages that ship none (sounddevice, pynvim,
faster-whisper); it is gitignored, so zanity does not check the declarations.
