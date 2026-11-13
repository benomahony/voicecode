"""The coding model, which rewrites the buffer as the spoken instructions ask."""


class CodeEditor:
    def __init__(self, model: str) -> None:
        from pydantic_ai import Agent

        assert model, "no coding model named; set --model or VOICECODE_MODEL"
        assert model == model.strip(), (
            f"coding model {model!r} has surrounding whitespace; check VOICECODE_MODEL"
        )
        self.agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            instructions=(
                "You are continuously editing ONE Neovim source buffer. "
                "The buffer contains ordinary source code and natural-language speech "
                "inserted by live transcription, usually as comments tagged 'voice:' "
                "(for example '# voice: add a timeout'). Transform the spoken instructions "
                "into working code and revise existing code as requested. Remove each "
                "voice comment once implemented, but keep every other comment. Preserve "
                "unrelated existing code. "
                "Return ONLY the complete replacement buffer contents, without fences, "
                "explanations, or commentary. Do not invent dependencies unless needed. "
                "If instructions are incomplete, preserve them until they can be implemented."
            ),
        )

    async def transform(self, content: str, filename: str) -> str:
        assert content.strip(), "asked to transform an empty buffer; snapshot() needs speech"
        assert filename, "no filename for the prompt; NvimEditor.name() falls back to untitled.py"
        result = await self.agent.run(
            f"Filename: {filename}\nCurrent entire buffer:\n<buffer>\n{content}\n</buffer>"
        )
        return result.output
