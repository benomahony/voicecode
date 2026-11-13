"""The coding model, which rewrites the code around the spoken instructions.

The model reads the whole buffer but writes back only a window of lines around the
speech: output tokens are what make a model slow, and a window is a small fraction of
a file.
"""

from pydantic import BaseModel

WINDOW_LINES = 20  # Lines kept on each side of the speech in the region the model rewrites.


class Edit(BaseModel):
    """The region, rewritten as the spoken instructions ask."""

    region: str  # The complete new text of the region, without fences or commentary.


def window(content: str, speech: range) -> range:
    """Whole lines from WINDOW_LINES above the speech to WINDOW_LINES below it, as
    character offsets; the last line's newline is left outside."""
    assert 0 <= speech.start <= speech.stop <= len(content), (
        f"speech {speech} is outside the {len(content)}-character buffer"
    )
    starts = [0] + [i + 1 for i, c in enumerate(content) if c == "\n"]
    first = max(0, content.count("\n", 0, speech.start) - WINDOW_LINES)
    last = min(len(starts) - 1, content.count("\n", 0, speech.stop) + WINDOW_LINES)
    begin = starts[first]
    end = starts[last + 1] - 1 if last + 1 < len(starts) else len(content)
    assert begin <= speech.start and speech.stop <= end, (
        f"window {begin}..{end} must contain the speech {speech}"
    )
    return range(begin, end)


class CodeEditor:
    def __init__(self, model: str) -> None:
        from pydantic_ai import Agent, ToolOutput

        assert model, "no coding model named; set --model or VOICECODE_MODEL"
        assert model == model.strip(), (
            f"coding model {model!r} has surrounding whitespace; check VOICECODE_MODEL"
        )
        self.agent: Agent[None, Edit] = Agent(
            model,
            # A tool and no free text, so prose can't be mistaken for code.
            output_type=ToolOutput(Edit, name="edit_region", description=Edit.__doc__),
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription, usually as comments tagged 'AI:' "
                "(for example '# AI: add a timeout'). Transform the spoken instructions "
                "into working code and revise existing code as requested. Remove each "
                "AI: comment once implemented, but keep every other comment. "
                "You are shown the whole buffer for context and one region of it, which "
                "holds the speech. Answer with edit_region: the complete new text of that "
                "region only, preserving its unrelated lines exactly. Do not invent "
                "dependencies unless needed. If instructions are incomplete, preserve them "
                "until they can be implemented."
            ),
        )

    async def transform(self, content: str, filename: str, speech: range) -> str:
        """The whole buffer with the region around `speech` rewritten by the model."""
        assert content[speech.start : speech.stop].strip(), "asked to transform without speech"
        assert filename, "no filename for the prompt; NvimEditor.name() falls back to untitled.py"
        region = window(content, speech)
        prompt = "\n".join(
            [
                f"Filename: {filename}",
                "Whole buffer, for context:",
                f"<buffer>\n{content}\n</buffer>",
                "Region to rewrite, which holds the speech:",
                f"<region>\n{content[region.start : region.stop]}\n</region>",
            ]
        )
        edit = (await self.agent.run(prompt)).output
        rewritten = edit.region.strip("\n")
        return content[: region.start] + rewritten + content[region.stop :]
