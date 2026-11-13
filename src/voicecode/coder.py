"""The coding model, which rewrites the buffer as the spoken instructions ask.

It can search the rest of the repository by description through code-chat's daemon,
so spoken references to existing code ("retry like the downloader does") resolve to
the real helpers instead of guessed names.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext

from code_chat.index import Hit

SEARCH_SECONDS = 30.0  # A busy or cold daemon past this, and the model carries on without.
SHOWN_HITS = 5
SHOWN_LINES = 40


@dataclass(frozen=True)
class Context:
    root: Path | None  # The git repository holding the buffer's file, if any.


def repository_of(filename: str) -> Path | None:
    """The git repository holding `filename`, or None for unnamed or loose files."""
    assert filename, "no filename; NvimEditor.name() falls back to untitled.py"
    path = Path(filename)
    if not path.is_absolute():
        return None  # An unnamed buffer: searching the backend's cwd would be wrong.
    root = next((p for p in path.parents if (p / ".git").exists()), None)
    assert root is None or path.is_relative_to(root), f"{root} doesn't contain {path}"
    return root


def describe(hits: list[Hit]) -> str:
    """Search hits as `path:start-end` headers over their code, for the model to read."""
    assert all(h["start"] <= h["end"] for h in hits), f"a hit ends before it starts: {hits}"
    blocks: list[str] = []
    for hit in hits[:SHOWN_HITS]:
        last = min(hit["end"], hit["start"] + SHOWN_LINES - 1)
        try:
            lines = Path(hit["file"]).read_text(errors="replace").split("\n")
            code = "\n".join(lines[hit["start"] - 1 : last])
        except OSError:
            code = hit["preview"]  # Deleted since it was indexed.
        blocks.append(f"{hit['path']}:{hit['start']}-{hit['end']}\n{code}")
    out = "\n\n".join(blocks) or "No matching code."
    assert len(blocks) <= SHOWN_HITS, f"showed {len(blocks)} hits, more than {SHOWN_HITS}"
    return out


async def search_code(ctx: RunContext[Context], description: str) -> str:
    """Find existing code in this repository that matches a description, such as
    "retry a failed download". Use it when the spoken instructions refer to code
    outside the buffer. Returns the best matches as path:start-end with their code."""
    from code_chat import daemon

    root = ctx.deps.root
    assert root is None or root.is_absolute(), f"repository root {root} must be absolute"
    if root is None:
        return "This buffer isn't in a git repository, so there is no other code to search."
    if not description.strip():
        return "Describe the code to find, for example 'retry a failed download'."
    print(f"Searching code for {description!r}", file=sys.stderr, flush=True)
    try:
        hits = await asyncio.to_thread(daemon.search_text, root, description, SEARCH_SECONDS)
    except (OSError, SystemExit) as error:
        # TimeoutError and ConnectionError are OSErrors; connect() exits if no daemon starts.
        return f"Code search is unavailable ({error}); write the code without it."
    assert all(Path(h["file"]).is_relative_to(root) for h in hits), f"hits outside {root}"
    return describe(hits)


class CodeEditor:
    def __init__(self, model: str) -> None:
        assert model, "no coding model named; set --model or VOICECODE_MODEL"
        assert model == model.strip(), (
            f"coding model {model!r} has surrounding whitespace; check VOICECODE_MODEL"
        )
        self.agent: Agent[Context, str] = Agent(
            model,
            deps_type=Context,
            output_type=str,
            tools=[search_code],
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription. Transform the spoken instructions into "
                "working code and revise existing code as requested. Remove natural-language "
                "instructions once implemented. Preserve unrelated existing code. "
                "When the instructions refer to existing code outside this buffer, call "
                "search_code with a short description and use what it finds instead of "
                "guessing names, imports or signatures. "
                "Return ONLY the complete replacement buffer contents, without fences, "
                "explanations, or commentary. Do not invent dependencies unless needed. "
                "If instructions are incomplete, preserve them until they can be implemented."
            ),
        )

    async def transform(self, content: str, filename: str) -> str:
        assert content.strip(), "asked to transform an empty buffer; snapshot() needs speech"
        assert filename, "no filename for the prompt; NvimEditor.name() falls back to untitled.py"
        result = await self.agent.run(
            f"Filename: {filename}\nCurrent entire buffer:\n<buffer>\n{content}\n</buffer>",
            deps=Context(repository_of(filename)),
        )
        return result.output
