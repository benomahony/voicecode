# voicecode

Live speech transcription and code transformation **in the same Neovim buffer**.
One Python process is the sole writer for speech and AI edits. The speech recognizer
and code agent never call Neovim directly.

## Install

Requires Python 3.12+, uv, Neovim, microphone permission, and an API key for the coding
model. The first run downloads the `base.en` Whisper model. On macOS, PortAudio
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

The default keymap is only created when the key is unmapped. Change or disable it
before the plugin loads:

```lua
vim.g.voicecode_keymap = "<leader>v" -- or false to disable
```

To override the backend command, set
`vim.g.voicecode_cmd = { "/path/to/voicecode" }`.

Set `OPENAI_API_KEY` for the default `openai:gpt-4.1-mini` Pydantic AI model.
`VOICECODE_MODEL` can select another Pydantic AI model. `VOICECODE_ASR_MODEL`
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

## Tests

```sh
uv run pytest -q
```
