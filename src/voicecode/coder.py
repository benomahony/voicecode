"""The coding model, which turns the spoken instructions into code.

The model reads the whole buffer and writes back only the lines it changes, as
search/replace blocks (the format aider uses): output tokens are what make a model
slow, and a spoken edit changes a few lines of a file.
"""

import re
from dataclasses import dataclass
from typing import cast

from pydantic_ai.models.openai import OpenAIChatModelSettings

SEARCH, DIVIDER, REPLACE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"
BLOCK = re.compile(rf"^{SEARCH}\n(.*?)^{DIVIDER}\n(.*?)^{REPLACE}$", flags=re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Change:
    search: str  # Lines exactly as they are in the buffer.
    replace: str  # What they become.


def changes(reply: str) -> list[Change]:
    """The search/replace blocks in the model's reply, in order."""
    found = [
        Change(cast(str, m.group(1)).removesuffix("\n"), cast(str, m.group(2)).removesuffix("\n"))
        for m in BLOCK.finditer(reply)
    ]
    assert len(found) <= reply.count(SEARCH), f"parsed {len(found)} blocks from fewer markers"
    assert all(SEARCH not in c.search for c in found), f"a block swallowed another: {found}"
    return found


def apply(content: str, edits: list[Change]) -> str:
    """`content` with each change made, in order; raises ValueError naming the first
    change whose search text isn't in the buffer exactly once."""
    assert edits, "apply() needs at least one change"
    for change in edits:
        count = content.count(change.search) if change.search else 0
        if count != 1:
            where = "isn't in the buffer" if count == 0 else f"is in the buffer {count} times"
            raise ValueError(f"the model's search text {change.search[:60]!r} {where}")
        content = content.replace(change.search, change.replace, 1)
    assert edits[-1].replace in content, f"{edits[-1].replace[:60]!r} is missing from the result"
    return content


class CodeEditor:
    def __init__(self, model: str) -> None:
        from pydantic_ai import Agent

        assert model, "no coding model named; set --model or VOICECODE_MODEL"
        assert model == model.strip(), (
            f"coding model {model!r} has surrounding whitespace; check VOICECODE_MODEL"
        )
        # OpenAI's current models reason before answering by default; a spoken edit
        # wants the answer at once, so turn reasoning off.
        self.settings: OpenAIChatModelSettings = (
            {"openai_reasoning_effort": "none"} if model.startswith("openai:") else {}
        )
        self.usage: str = ""  # Token counts of the last edit, for the log.
        self.failure: str = ""  # Why the last reply couldn't be applied, for the log.
        self.reply: str = ""  # The model's last reply, as it gave it.
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
                "Answer with ONLY search/replace blocks, nothing else:\n"
                f"{SEARCH}\n<lines exactly as they are in the buffer>\n{DIVIDER}\n"
                f"<what they become>\n{REPLACE}\n"
                "Each search must match the buffer exactly, whitespace included, and only "
                "once: include enough lines to make it unique, but no more than needed. "
                "Include the AI: comment lines in the search so they are replaced. Do not "
                "invent dependencies unless needed. If the instructions are incomplete, "
                "answer with no blocks and they stay until they can be implemented."
            ),
        )

    async def transform(self, content: str, filename: str) -> str:
        """The whole buffer with the model's changes made, or "" (with `failure` set)
        when there are none or they don't fit the buffer."""
        assert content.strip(), "asked to transform an empty buffer; snapshot() needs speech"
        assert filename, "no filename for the prompt; NvimEditor.name() falls back to untitled.py"
        prompt = f"Filename: {filename}\n<buffer>\n{content}\n</buffer>"
        result = await self.agent.run(prompt, model_settings=self.settings)
        used = result.usage
        self.usage = f"{used.input_tokens} tokens in, {used.output_tokens} out"
        self.reply = result.output
        edits = changes(result.output)
        if not edits:
            self.failure = "the model made no changes"
            return ""
        try:
            return apply(content, edits)
        except ValueError as error:
            self.failure = str(error)
            return ""
