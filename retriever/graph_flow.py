from typing import List, Dict, Literal
from langgraph.graph import StateGraph, START, END
from models import (
    InputState,
    OutputState,
    OverallState,
    Node,
    InitialNodes,
    AtomicFactOutput,
    ChunkOutput,
    NeighborOutput,
    AnswerReasonOutput,
    CustomChatLLM,
    MyServerEmbeddings,
)
import prompts
from utils import parse_function
import config
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize Neo4j graph
graph = Neo4jGraph(
    refresh_schema=False,
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    database=config.NEO4J_DATABASE
)

# Initialize custom LLM
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE,
)

llm = CustomChatLLM(
    client=client,
    model_name=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
    ignore_eos=False,
    stop_tokens=[config.EOS_TOKEN],
    bos_token=config.BOS_TOKEN,
    eos_token=config.EOS_TOKEN,
    add_generation_prompt=True,
)

# Initialize embeddings
embeddings_local = MyServerEmbeddings(server_url=config.EMBEDDINGS_SERVER_URL)

neo4j_vector = Neo4jVector.from_existing_graph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    graph=graph,
    embedding=embeddings_local,
    index_name="keyelements",
    node_label="KeyElement",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    retrieval_query="RETURN node.id AS text, score, {} AS metadata"
)

# Utility functions
def get_potential_nodes(question: str) -> List[str]:
    data = neo4j_vector.similarity_search(question, k=20)
    return [el.page_content for el in data]

def get_atomic_facts(key_elements: List[str]) -> List[Dict[str, str]]:
    data = graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-(fact)<-[:HAS_ATOMIC_FACT]-(chunk)
    WHERE k.id IN $key_elements
    RETURN distinct chunk.id AS chunk_id, fact.text AS text
    """, params={"key_elements": key_elements})
    return data

def get_neighbors_by_key_element(key_elements: List[str]) -> List[str]:
    data = graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-()-[:HAS_KEY_ELEMENT]->(neighbor)
    WHERE k.id IN $key_elements AND NOT neighbor.id IN $key_elements
    WITH neighbor, count(*) AS count
    ORDER BY count DESC LIMIT 50
    RETURN collect(neighbor.id) AS possible_candidates
    """, params={"key_elements": key_elements})
    return data[0]['possible_candidates'] if data else []

def get_chunk(chunk_id: str) -> List[Dict[str, str]]:
    data = graph.query("""
    MATCH (c:Chunk)
    WHERE c.id = $chunk_id
    RETURN c.id AS chunk_id, c.text AS text
    """, params={"chunk_id": chunk_id})
    return data

def get_subsequent_chunk_id(chunk_id: str) -> str:
    data = graph.query("""
    MATCH (c:Chunk)-[:NEXT]->(next)
    WHERE c.id = $chunk_id
    RETURN next.id AS next
    """, params={"chunk_id": chunk_id})
    return data[0]['next'] if data else None

def get_previous_chunk_id(chunk_id: str) -> str:
    data = graph.query("""
    MATCH (c:Chunk)<-[:NEXT]-(previous)
    WHERE c.id = $chunk_id
    RETURN previous.id AS previous
    """, params={"chunk_id": chunk_id})
    return data[0]['previous'] if data else None

# Define prompts and chains
# [Include your prompt templates and chains here, similar to your original code]

###--------------------------------------------------------------------------------------
# For brevity, here's an example of defining one chain
rational_plan_system = prompts.rational_plan_system
rational_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(rational_plan_system),
        HumanMessagePromptTemplate.from_template('{question}'),
    ]
)
rational_chain = rational_prompt | llm | StrOutputParser()
###--------------------------------------------------------------------------------------
initial_node_system = prompts.initial_node_system

initial_node_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(initial_node_system),
        HumanMessagePromptTemplate.from_template("""Вопрос: {question}
                                                    План: {rational_plan}
                                                    Узлы: {nodes}
                                                    """),
    ]
)
parser_initial_nodes = PydanticOutputParser(pydantic_object=InitialNodes)
initial_nodes_chain = initial_node_prompt | llm | parser_initial_nodes
###--------------------------------------------------------------------------------------
atomic_fact_check_system = prompts.atomic_fact_check_system

parser_atomic_fact_check = PydanticOutputParser(pydantic_object=AtomicFactOutput)
atomic_fact_check_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(atomic_fact_check_system),
        HumanMessagePromptTemplate.from_template("""Вопрос: {question}
                                                    План: {rational_plan}
                                                    Предыдущее действие: {previous_actions}
                                                    Тетрадка: {notebook}
                                                    Атомарный факт: {atomic_facts}""")
    ]
)
atomic_fact_chain = atomic_fact_check_prompt | llm | parser_atomic_fact_check
# Define other chains similarly...
###--------------------------------------------------------------------------------------
chunk_read_system = prompts.chunk_read_system

parser_chunk_read = PydanticOutputParser(pydantic_object=ChunkOutput)
chunk_read_prompt = ChatPromptTemplate.from_messages(
    [
       SystemMessagePromptTemplate.from_template(chunk_read_system),
       HumanMessagePromptTemplate.from_template("""Вопрос: {question}
                                                План: {rational_plan}
                                                Предыдущие действия: {previous_actions}
                                                Тетрадка: {notebook}
                                                Текстовый фрагмент: {chunk}""")
    ]
)
chunk_read_chain = chunk_read_prompt | llm | parser_chunk_read
###--------------------------------------------------------------------------------------
neighbor_select_system = prompts.neighbor_select_system

parser_neighbour_select = PydanticOutputParser(pydantic_object=NeighborOutput)
neighbor_select_prompt = ChatPromptTemplate.from_messages(
    [
       SystemMessagePromptTemplate.from_template(neighbor_select_system),
       HumanMessagePromptTemplate.from_template("""Вопрос: {question}
                                                План: {rational_plan}
                                                Предыдущее действие: {previous_actions}
                                                Тетрадка: {notebook}
                                                Соседние узлы: {nodes}""")
    ]
)
neighbor_select_chain = neighbor_select_prompt | llm | parser_neighbour_select
###--------------------------------------------------------------------------------------
answer_reasoning_system = prompts.answer_reasoning_system

parser_answer_reasoning = PydanticOutputParser(pydantic_object=AnswerReasonOutput)
answer_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        
        SystemMessagePromptTemplate.from_template(answer_reasoning_system),
        HumanMessagePromptTemplate.from_template("""Вопрос: {question}
                                                    Тетрадка: {notebook}
                                                 """),
    ]
)
answer_reasoning_chain = answer_reasoning_prompt | llm | parser_answer_reasoning
###--------------------------------------------------------------------------------------



# Node functions
def rational_plan_node(state: InputState) -> OverallState:
    rational_plan = rational_chain.invoke({"question": state.get("question")})
    print("-" * 20)
    print(f"Step: rational_plan")
    print(f"Rational plan: {rational_plan}")
    return {
        "rational_plan": rational_plan,
        "previous_actions": ["rational_plan"],
    }

def initial_node_selection(state: OverallState) -> OverallState:
    potential_nodes = get_potential_nodes(state.get("question"))
    print(potential_nodes)
    initial_nodes = initial_nodes_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "nodes": potential_nodes,
            "format_instructions": parser_initial_nodes.get_format_instructions()
        }
    )  
    # paper uses 5 initial nodes
    check_atomic_facts_queue = [
        el.key_element
        for el in sorted(
            initial_nodes.initial_nodes,
            key=lambda node: node.score,
            reverse=True,
        )
    ][:10]
    print(check_atomic_facts_queue)
    return {
        "check_atomic_facts_queue": check_atomic_facts_queue,
        "previous_actions": ["initial_node_selection"],
    }

def atomic_fact_check(state: OverallState) -> OverallState:
    atomic_facts = get_atomic_facts(state.get("check_atomic_facts_queue"))
    print(f"Atomic Facts: {atomic_facts}")
    print("-" * 20)
    print(f"Step: atomic_fact_check")
    print(
        f"Reading atomic facts about: {state.get('check_atomic_facts_queue')}"
    )
    atomic_facts_results = atomic_fact_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "atomic_facts": atomic_facts,
            "format_instructions": parser_atomic_fact_check.get_format_instructions()
        }
    )

    notebook = atomic_facts_results.updated_notebook
    print(
        f"Rational for next action after atomic check: {atomic_facts_results.rational_next_action}"
    )
    chosen_action = parse_function(atomic_facts_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    print(f"updated notebook: {notebook}")

    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "check_atomic_facts_queue": [],
        "previous_actions": [
            f"atomic_fact_check({state.get('check_atomic_facts_queue')})"
        ],
    }
    if chosen_action.get("function_name") == "stop_and_read_neighbor":
        neighbors = get_neighbors_by_key_element(
            state.get("check_atomic_facts_queue")
        )
        print(f"Cypher response get_neighbors_by_key_element: {neighbors}")
        response["neighbor_check_queue"] = neighbors
    elif chosen_action.get("function_name") == "read_chunk":
        response["check_chunks_queue"] = chosen_action.get("arguments")[0]
    return response

def chunk_check(state: OverallState) -> OverallState:
    check_chunks_queue = state.get("check_chunks_queue")
    chunk_id = check_chunks_queue.pop()
    print("-" * 20)
    print(f"Step: read chunk({chunk_id})")

    chunks_text = get_chunk(chunk_id)
    read_chunk_results = chunk_read_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "chunk": chunks_text,
            "format_instructions": parser_chunk_read.get_format_instructions()
        }
    )

    notebook = read_chunk_results.updated_notebook
    print(
        f"Rational for next action after reading chunks: {read_chunk_results.rational_next_move}"
    )
    chosen_action = parse_function(read_chunk_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    print(f"updated notebook: {notebook}")

    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "previous_actions": [f"read_chunks({chunk_id})"],
    }
    if chosen_action.get("function_name") == "read_subsequent_chunk":
        subsequent_id = get_subsequent_chunk_id(chunk_id)
        check_chunks_queue.append(subsequent_id)
    elif chosen_action.get("function_name") == "read_previous_chunk":
        previous_id = get_previous_chunk_id(chunk_id)
        check_chunks_queue.append(previous_id)
    elif chosen_action.get("function_name") == "search_more":
        # Go over to next chunk
        # Else explore neighbors
        if not check_chunks_queue:
            response["chosen_action"] = "search_neighbor"
            # Get neighbors/use vector similarity
            print(f"Neighbor rational: {read_chunk_results.rational_next_move}")
            neighbors = get_potential_nodes(
                read_chunk_results.rational_next_move
            )
            response["neighbor_check_queue"] = neighbors

    response["check_chunks_queue"] = check_chunks_queue
    return response

def neighbor_select(state: OverallState) -> OverallState:
    print("-" * 20)
    print(f"Step: neighbor select")
    print(f"Possible candidates: {state.get('neighbor_check_queue')}")
    neighbor_select_results = neighbor_select_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "nodes": state.get("neighbor_check_queue"),
            "previous_actions": state.get("previous_actions"),
            "format_instructions": parser_neighbour_select.get_format_instructions()
        }
    )
    print(
        f"Rational for next action after selecting neighbor: {neighbor_select_results.rational_next_move}"
    )
    chosen_action = parse_function(neighbor_select_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    # Empty neighbor select queue
    response = {
        "chosen_action": chosen_action.get("function_name"),
        "neighbor_check_queue": [],
        "previous_actions": [
            f"neighbor_select({chosen_action.get('arguments', [''])[0] if chosen_action.get('arguments', ['']) else ''})"
        ],
    }
    if chosen_action.get("function_name") == "read_neighbor_node":
        response["check_atomic_facts_queue"] = [
            chosen_action.get("arguments")[0]
        ]
    return response

def answer_reasoning(state: OverallState) -> OutputState:
    print("-" * 20)
    print("Step: Answer Reasoning")
    final_answer = answer_reasoning_chain.invoke(
        {"question": state.get("question"), 
         "notebook": state.get("notebook"),
         "format_instructions": parser_answer_reasoning.get_format_instructions()
         }
    )
    return {
        "answer": final_answer.final_answer,
        "analysis": final_answer.analyze,
        "previous_actions": ["answer_reasoning"],
    }

# Define other node functions similarly...

# Conditions
def atomic_fact_condition(
    state: OverallState,
) -> Literal["neighbor_select", "chunk_check"]:
    if state.get("chosen_action") == "stop_and_read_neighbor":
        return "neighbor_select"
    elif state.get("chosen_action") == "read_chunk":
        return "chunk_check"

def chunk_condition(
    state: OverallState,
) -> Literal["answer_reasoning", "chunk_check", "neighbor_select"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") in ["read_subsequent_chunk", "read_previous_chunk", "search_more"]:
        return "chunk_check"
    elif state.get("chosen_action") == "search_neighbor":
        return "neighbor_select"

def neighbor_condition(
    state: OverallState,
) -> Literal["answer_reasoning", "atomic_fact_check"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") == "read_neighbor_node":
        return "atomic_fact_check"

# Build the LangGraph
def build_langgraph():
    langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
    langgraph.add_node(rational_plan_node)
    langgraph.add_node(initial_node_selection)
    langgraph.add_node(atomic_fact_check)
    langgraph.add_node(chunk_check)
    langgraph.add_node(answer_reasoning)
    langgraph.add_node(neighbor_select)

    langgraph.add_edge(START, "rational_plan_node")
    langgraph.add_edge("rational_plan_node", "initial_node_selection")
    langgraph.add_edge("initial_node_selection", "atomic_fact_check")
    langgraph.add_conditional_edges(
        "atomic_fact_check",
        atomic_fact_condition,
    )
    langgraph.add_conditional_edges(
        "chunk_check",
        chunk_condition,
    )
    langgraph.add_conditional_edges(
        "neighbor_select",
        neighbor_condition,
    )
    langgraph.add_edge("answer_reasoning", END)

    langgraph = langgraph.compile()
    return langgraph