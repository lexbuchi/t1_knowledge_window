# process_document.py

import asyncio
from datetime import datetime
from langchain.text_splitter import TokenTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain_community.graphs import Neo4jGraph
from models import Extraction
from llm import CustomLLM
from utils import construct_prompt, encode_md5
import config

# Initialize Neo4j graph
graph = Neo4jGraph(
    refresh_schema=False,
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    database=config.NEO4J_DATABASE
)

# Ensure constraints are set
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

# Initialize LLM
llm = CustomLLM(
    client=config.get_openai_client(),
    model=config.MODEL_NAME,
    max_tokens=config.MAX_TOKENS,
    temperature=config.TEMPERATURE,
    ignore_eos=False,
    stop_tokens=[config.EOS_TOKEN]
)

# Initialize prompt parser
parser = PydanticOutputParser(pydantic_object=Extraction)
parser_format_instructions = parser.get_format_instructions()

# Main processing function
async def process_document(text, document_name, chunk_size=250, chunk_overlap=30):
    start = datetime.now()
    print(f"Started extraction at: {start}")

    # Text splitting
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    print(f"Total text chunks: {len(texts)}")

    tasks = []
    for index, chunk_text in enumerate(texts):
        prompt_text = construct_prompt(
            template_jinja=config.TEMPLATE_JINJA,
            bos_token=config.BOS_TOKEN,
            eos_token=config.EOS_TOKEN,
            task_instruction=config.TASK_INSTRUCTION_ENG,
            parser_instruction=parser_format_instructions,
            additional_inst=config.ADDITIONAL_INST_ENG,
            pre_query=config.PRE_QUERY_ENG,
            query=chunk_text
        )
        tasks.append(llm.apredict(prompt_text))

    responses = await asyncio.gather(*tasks)
    print(f"Finished LLM extraction after: {datetime.now() - start}")

    parsed_outputs = []
    errored_indices = []
    for index, response in enumerate(responses):
        try:
            parsed_output = parser.parse(response)
            parsed_outputs.append((index, parsed_output))
        except Exception as e:
            print(f"Error parsing chunk at index {index}: {e}")
            print(f"Model output was: {response}")
            errored_indices.append(index)

    # Inform about the parsing results
    total_chunks = len(texts)
    successful_chunks = len(parsed_outputs)
    failed_chunks = len(errored_indices)
    print(f"Total chunks processed: {total_chunks}")
    print(f"Chunks successfully parsed: {successful_chunks}")
    print(f"Chunks failed to parse: {failed_chunks}")
    if failed_chunks > 0:
        print(f"Indices of failed chunks: {errored_indices}")

    # Process the successfully parsed outputs
    docs = []
    for index, parsed_output in parsed_outputs:
        doc = parsed_output.dict()
        chunk_text = texts[index]
        doc['chunk_id'] = encode_md5(chunk_text)
        doc['chunk_text'] = chunk_text
        doc['index'] = index
        for af in doc["atomic_facts"]:
            af["id"] = encode_md5(af["atomic_fact"])
        docs.append(doc)

    # Import chunks/atomic facts/key elements into the graph
    if docs:
        graph.query(config.IMPORT_QUERY, params={"data": docs, "document_name": document_name})

        # Create next relationships between chunks
        graph.query(config.CREATE_NEXT_QUERY, params={"document_name": document_name})

    print(f"Finished import at: {datetime.now() - start}")