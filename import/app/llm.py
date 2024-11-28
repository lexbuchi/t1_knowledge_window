# llm.py

from typing import Any, List, Optional
from pydantic import Field
from langchain.llms.base import LLM
import asyncio

class CustomLLM(LLM):
    client: Any
    model: str = Field(default="kg-wizard", description="The model to be used for generation")
    max_tokens: int = Field(default=2500, description="The maximum number of tokens to generate")
    temperature: float = Field(default=0.0, description="The temperature for controlling randomness in generation")
    ignore_eos: bool = Field(default=False, description="Whether to ignore end of sequence tokens")
    stop_tokens: List[str] = Field(default_factory=lambda: ["</s>"], description="List of stop tokens")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={
                "ignore_eos": self.ignore_eos,
                "stop": stop or self.stop_tokens,
            }
        )
        return completion.choices[0].text.strip()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._call, prompt, stop)
        return result

    @property
    def _llm_type(self) -> str:
        return "custom"

    async def apredict(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return await self._acall(prompt, stop=stop)