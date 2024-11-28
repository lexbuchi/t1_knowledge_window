from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict, TypedDict, Annotated
import asyncio
from abc import ABC
import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration
)
from openai import OpenAI
import config
from langgraph.graph.message import add_messages


# Custom LLM class
class CustomChatLLM(BaseChatModel):
    client: Any
    model_name: str = Field(default=config.MODEL_NAME, description="The model to be used for generation")
    max_tokens: int = Field(default=config.MAX_TOKENS, description="The maximum number of tokens to generate")
    temperature: float = Field(default=config.TEMPERATURE, description="The temperature for controlling randomness in generation")
    ignore_eos: bool = Field(default=False, description="Whether to ignore end of sequence tokens")
    stop_tokens: List[str] = Field(default_factory=lambda: [config.EOS_TOKEN], description="List of stop tokens")
    bos_token: str = Field(default=config.BOS_TOKEN, description="Beginning of sequence token")
    eos_token: str = Field(default=config.EOS_TOKEN, description="End of sequence token")
    add_generation_prompt: bool = Field(default=True, description="Whether to add generation prompt")
    top_k: int = Field(default=40, description="The number of top-k tokens to consider for generation")

    def _construct_prompt(self, messages: List[BaseMessage]) -> str:
        from jinja2 import Template

        template_jinja = """
{{ bos_token }}{% for message in messages %}
<|start_header_id|>{{ message.type }}<|end_header_id|>

{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
        jinja_template = Template(template_jinja)
        rendered_prompt = jinja_template.render(
            messages=messages,
            bos_token=self.bos_token,
            add_generation_prompt=self.add_generation_prompt
        )
        return rendered_prompt

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={
                "ignore_eos": self.ignore_eos,
                "stop": stop or self.stop_tokens,
                "top_k": self.top_k  # Add top_k here
            }
        )
        return completion.choices[0].text.strip()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._call, prompt, stop)
        return result

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        prompt = self._construct_prompt(messages)
        response_text = self._call(prompt, stop=stop)
        message = AIMessage(content=response_text)
        generation = ChatGeneration(text=response_text, message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None) -> List[ChatResult]:
        results = []
        for message_list in messages:
            prompt = self._construct_prompt(message_list)
            response_text = await self._acall(prompt, stop=stop)
            message = AIMessage(content=response_text)
            generation = ChatGeneration(text=response_text, message=message)
            results.append(ChatResult(generations=[generation]))
        return results

    @property
    def _llm_type(self) -> str:
        return "custom_chat_model"

# Custom Embeddings class
class MyServerEmbeddings(ABC):
    """Custom Embeddings class for embeddings served on a custom server."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(f"{self.server_url}/predict", json={"text": text})
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.server_url}/predict", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        from langchain_core.runnables.config import run_in_executor
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        from langchain_core.runnables.config import run_in_executor
        return await run_in_executor(None, self.embed_query, text)

# Data models
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
    analysis: str
    previous_actions: List[str]

class OverallState(TypedDict):
    question: str
    rational_plan: str
    notebook: str
    previous_actions: Annotated[List[str], add_messages]
    check_atomic_facts_queue: List[str]
    check_chunks_queue: List[str]
    neighbor_check_queue: List[str]
    chosen_action: str

# Additional Pydantic models
class Node(BaseModel):
    key_element: str = Field(description="""Ключевой элемент или название релевантного узла""")
    score: int = Field(description="""Релевантность возможному ответу, оцененная в диапазоне от 0 до 100. Оценка 100 указывает на высокую вероятность релевантности, тогда как оценка 0 указывает на минимальную релевантность.""")

class InitialNodes(BaseModel):
    initial_nodes: List[Node] = Field(description="Список релевантных узлов для вопроса и плана")

# class AtomicFactOutput(BaseModel):
#     updated_notebook: str = Field(description="""Сначала объедините вашу текущую тетрадку с новыми выводами и находками о вопросе из текущих атомарных фактов, создавая более полную версию тетрадки, содержащую более достоверную информацию.""")
#     rational_next_action: str = Field(description="""На основе данного вопроса, рационального плана, предыдущих действий и содержимого тетрадки проанализируйте, как выбрать следующее действие.""")
#     chosen_action: str = Field(description="""
# 1. read_chunk(List[ID]): Выберите это действие, если считаете, что текстовый фрагмент, связанный с атомарным фактом, может содержать необходимую информацию для ответа на вопрос. Это позволит получить более полную и детализированную информацию. Например, 'read_chunk([\"id1\", \"id2\"])'. Замените [\"id1\", \"id2\"] на фактический список идентификаторов фрагментов, которые вы хотите прочитать.",
# 2. stop_and_read_neighbor(): Выберите это действие, если полагаете, что все текстовые фрагменты не содержат полезной информации.""")
class AtomicFactOutput(BaseModel):
    updated_notebook: str = Field(description="""First, combine your current notebook with new insights and findings about
the question from current atomic facts, creating a more complete version of the notebook that
contains more valid information.""")
    rational_next_action: str = Field(description="""Based on the given question, the rational plan, previous actions, and
notebook content, analyze how to choose the next action.""")
    chosen_action: str = Field(description="""1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic
fact may hold the necessary information to answer the question. This will allow you to access
more complete and detailed information. For example 'read_chunk([\"id1\", \"id2\"])'. Replace [\"id1\", \"id2\"] with the actual list of chunk IDs you want to read.",
2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable
information.""")

class ChunkOutput(BaseModel):
    updated_notebook: str = Field(description="""Сначала объедините предыдущие записи с новыми выводами и находками о вопросе из текущих текстовых фрагментов, создавая более полную версию записной книжки, содержащую более достоверную информацию.""")
    rational_next_move: str = Field(description="""На основе данного вопроса, рационального плана, предыдущих действий и содержимого записной книжки проанализируйте, как выбрать следующее действие.""")
    chosen_action: str = Field(description="""1. search_more(): Выберите это действие, если считаете, что необходимой информации для ответа на вопрос все еще недостаточно.
2. read_previous_chunk(): Выберите это действие, если считаете, что предыдущий текстовый фрагмент содержит полезную информацию для ответа на вопрос.
3. read_subsequent_chunk(): Выберите это действие, если считаете, что последующий текстовый фрагмент содержит полезную информацию для ответа на вопрос.
4. termination(): Выберите это действие, если считаете, что имеющейся информации достаточно для ответа на вопрос. Это позволит вам обобщить собранную информацию и предоставить окончательный ответ.""")


class NeighborOutput(BaseModel):
    rational_next_move: str = Field(description="""На основе данного вопроса, рационального плана, предыдущих действий и
содержимого тетрадки проанализируйте, как выбрать следующее действие.""")
    chosen_action: str = Field(description="""У вас есть следующие варианты действий:
1. read_neighbor_node(key element of node): Выберите это действие, если считаете, что любой из
соседних узлов может содержать информацию, релевантную вопросу. Обратите внимание, что следует
сфокусироваться только на одном соседнем узле за раз. `key element of node` – это название узла. Нужно выбрать один за раз из списка соседних узлов. Например, если список Соседних узлов: ['название_1', 'название_2', 'название_3'], то нужно выбрать один из самых подходящих элементов списка, и вызвать функцию, например: read_neighbor_node(название_1)
2. termination(): Выберите это действие, если считаете, что ни один из соседних узлов не содержит
информации, которая могла бы ответить на вопрос.""")

class AnswerReasonOutput(BaseModel):
    analyze: str = Field(description="""Сначала проанализируйте содержание каждой записной книжки, прежде чем дать окончательный ответ.
Во время анализа учитывайте дополнительную информацию из других записей и используйте стратегию
большинства для разрешения любых несоответствий.""")
    final_answer: str = Field(description="""При создании окончательного ответа учитывайте всю доступную информацию.""")
# class AnswerReasonOutput(BaseModel):
#     analyze: str = Field(description="""You should first analyze each notebook content before providing a final answer.
#     During the analysis, consider complementary information from other notes and employ a
# majority voting strategy to resolve any inconsistencies.""")
#     final_answer: str = Field(description="""When generating the final answer, ensure that you take into account all available information.""")