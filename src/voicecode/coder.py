"""The coding model, which turns the spoken instructions into code.

The model reads the whole buffer and writes back only the lines it changes, as
search/replace blocks (the format aider uses): output tokens are what make a model
slow, and a spoken edit changes a few lines of a file.
"""

import re
from dataclasses import dataclass, replace
from typing import cast

from pydantic_ai import Conversation
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModelSettings

MODEL = "openai:gpt-6-luna"  # The default; VOICECODE_MODEL picks another.
SEARCH, DIVIDER, REPLACE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"
BLOCK = re.compile(rf"^{SEARCH}\n(.*?)^{DIVIDER}\n(.*?)^{REPLACE}$", flags=re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Kind:
    """What sort of file the buffer is, so the model writes in its syntax."""

    filetype: str  # Neovim's 'filetype', e.g. "asciidoc"; "" when unknown.
    comment: tuple[str, str]  # Its comment opener and closer, e.g. ("//", "").


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


def echoed(edits: list[Change]) -> list[str]:
    """Non-blank lines a reply sends back unchanged: in a block's search and its replace."""
    repeated: list[str] = []
    for edit in edits:
        kept = set(edit.search.split("\n"))
        repeated += [line for line in edit.replace.split("\n") if line.strip() and line in kept]
    assert all(line.strip() for line in repeated), f"blank lines aren't echoes: {repeated}"
    assert len(repeated) <= sum(e.replace.count("\n") + 1 for e in edits), f"{repeated}"
    return repeated


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


RECENT = 6  # Messages of conversation kept: the last three edits and their replies.
OLD_BUFFER = re.compile(r"<buffer>\n.*?\n</buffer>", flags=re.DOTALL)
STALE = "<buffer>\n(an earlier copy, now out of date; the latest message has the buffer)\n</buffer>"


def recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The last RECENT messages, starting at a request, with every buffer but the newest
    cut out: only the latest copy is current, and old copies would cost a file per edit."""
    kept = messages[-RECENT:]
    while kept and not isinstance(kept[0], ModelRequest):
        kept = kept[1:]
    trimmed = [stale(m) if i < len(kept) - 1 else m for i, m in enumerate(kept)]
    assert trimmed and isinstance(trimmed[0], ModelRequest), (
        f"history must start with a request, not {type(trimmed[0]).__name__ if trimmed else None}"
    )
    assert len(trimmed) <= RECENT, f"kept {len(trimmed)} messages, more than {RECENT}"
    return trimmed


def stale(message: ModelMessage) -> ModelMessage:
    """A request with its buffer copy replaced by a note that it's out of date."""
    if not isinstance(message, ModelRequest):
        return message
    assert message.parts, "a request without parts can't be sent; drop it instead"
    parts = [
        replace(part, content=OLD_BUFFER.sub(STALE, part.content))
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        else part
        for part in message.parts
    ]
    assert len(parts) == len(message.parts), f"{len(message.parts)} parts became {len(parts)}"
    return replace(message, parts=parts)


class CodeEditor:
    def __init__(self, model: str) -> None:
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import Caching, ProcessHistory

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
        self.conversations: dict[str, Conversation] = {}  # One per file being dictated into.
        self.agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            capabilities=[Caching(), ProcessHistory(recent)],
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription, usually as comments tagged 'AI:' in the "
                "file's own comment syntax, which the prompt names. Transform the spoken "
                "instructions into working code and revise existing code as requested. "
                "Remove each AI: comment once implemented, but keep every other comment, "
                "and never change the syntax of existing comments. "
                "Speech is edited in continuously, so an earlier edit may already have "
                "turned the start of a sentence into code, leaving only its end in an AI: "
                "comment. Earlier messages show what was said and how you answered: read "
                "the AI: comments as continuing them, and extend the code you wrote rather "
                "than treating the rest of a sentence as a new request. Earlier replies may "
                "not have been applied; the buffer in the latest message is the current one. "
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

    async def transform(self, content: str, filename: str, kind: Kind) -> str:
        """The whole buffer with the model's changes made, or "" (with `failure` set)
        when there are none or they don't fit the buffer."""
        assert content.strip(), "asked to transform an empty buffer; snapshot() needs speech"
        assert filename, "no filename for the prompt; NvimEditor.name() falls back to untitled.py"
        opener, closer = kind.comment
        syntax = f"{opener} AI: ... {closer}".strip() if opener else "plain text, no comments"
        prompt = "\n".join(
            [
                f"Filename: {filename}",
                f"Filetype: {kind.filetype or 'unknown'}",
                f"Spoken instructions in this file look like: {syntax}",
                f"<buffer>\n{content}\n</buffer>",
            ]
        )
        try:
            result = await self.agent.run(
                prompt, model_settings=self.settings, conversation=self.conversations.get(filename)
            )
        except UnexpectedModelBehavior as error:
            # An empty reply, retried until the limit: the model had nothing to change.
            self.failure = f"the model gave no usable reply ({error})"
            return ""
        self.conversations[filename] = result.conversation
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
