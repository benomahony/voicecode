"""The coding model, which rewrites the code around the spoken instructions.

The model reads the whole buffer but writes back only a window of lines around the
speech: output tokens are what make a model slow, and a window is a small fraction of
a file.
"""

from pydantic_ai.models.openai import OpenAIChatModelSettings

WINDOW_LINES = 20  # Lines kept on each side of the speech in the region the model rewrites.
FENCE = "```"


def unfenced(text: str) -> str:
    """The model's region without a Markdown code fence around it, if it added one."""
    lines = text.strip("\n").split("\n")
    fenced = len(lines) >= 2 and lines[0].startswith(FENCE) and lines[-1].strip() == FENCE
    if fenced:
        lines = lines[1:-1]
    out = "\n".join(lines)
    assert len(out) <= len(text), f"unfencing grew {len(text)} characters to {len(out)}"
    assert not fenced or not out.endswith(FENCE), f"left the closing fence in {out[-60:]!r}"
    return out


def resembles(rewritten: str, region: str) -> bool:
    """Whether a rewritten region still shares a line with the original; a reply with
    none is prose, not the region, and must not replace code."""
    kept = {stripped for stripped in map(str.strip, region.split("\n")) if stripped}
    new = set(map(str.strip, rewritten.split("\n")))
    shared = bool(kept & new)
    assert kept or not shared, f"shared {kept & new} with an empty region"
    assert not shared or rewritten.strip(), f"empty rewrite {rewritten!r} shared {kept & new}"
    return shared or not kept


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
        from pydantic_ai import Agent

        assert model, "no coding model named; set --model or VOICECODE_MODEL"
        assert model == model.strip(), (
            f"coding model {model!r} has surrounding whitespace; check VOICECODE_MODEL"
        )
        # Plain text, not a tool call, so OpenAI's predicted outputs can stream back the
        # unchanged lines of the region almost for free; tools would rule that out.
        self.predict: bool = model.startswith("openai:")
        self.agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription, usually as comments tagged 'AI:' "
                "(for example '# AI: add a timeout'). Transform the spoken instructions "
                "into working code and revise existing code as requested. Remove each "
                "AI: comment once implemented, but keep every other comment. "
                "You are shown the whole buffer for context and one region of it, which "
                "holds the speech. Answer with ONLY the complete new text of that region: "
                "no fences, explanations or commentary, and its unrelated lines exactly as "
                "they were. Do not invent "
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
        original = content[region.start : region.stop]
        settings: OpenAIChatModelSettings = (
            {"openai_prediction": {"type": "content", "content": original}} if self.predict else {}
        )
        rewritten = unfenced((await self.agent.run(prompt, model_settings=settings)).output)
        if not resembles(rewritten, original):
            return ""  # Prose: an empty edit, which apply_code refuses.
        return content[: region.start] + rewritten + content[region.stop :]
