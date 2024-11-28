# config.py

import os
from openai import OpenAI

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://0.0.0.0:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "<neo4j_user>")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "<password>")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "<neo4j_database>")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8004/v1")

# Model configuration
BOS_TOKEN = '<|begin_of_text|>'
EOS_TOKEN = '<|end_of_text|>'
MODEL_NAME = "vikhr_llama"
TEMPERATURE = 0.0
MAX_TOKENS = 2500

# Prompt and task configuration
TEMPLATE_JINJA = """
{{ bos_token }}{% for message in messages %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
TASK_INSTRUCTION_ENG = """
You are now an intelligent assistant tasked with meticulously extracting both key elements and
atomic facts from a long text.
1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the text’s narrative.
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include
propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.
Requirements:
#####
1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
2. You should extract key elements and atomic facts comprehensively, especially those that are
important and potentially query-worthy and do not leave out details.
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He,
She to actual names).
4. Ensure that the key elements and atomic facts you extract are presented in the same language as
the original text (e.g., English or Chinese).
"""
ADDITIONAL_INST_ENG = 'Write ONLY JSON without any text or format elements'
PRE_QUERY_ENG = 'Use the given format to extract information from the following input:'

# Import query for Neo4j
IMPORT_QUERY = """
MERGE (d:Document {id:$document_name})
WITH d
UNWIND $data AS row
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.chunk_text,
    c.index = row.index,
    c.document_name = row.document_name
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.atomic_facts AS af
MERGE (a:AtomicFact {id: af.id})
SET a.text = af.atomic_fact
MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
WITH c, a, af
UNWIND af.key_elements AS ke
MERGE (k:KeyElement {id: ke})
MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
"""
CREATE_NEXT_QUERY = """
MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
WHERE d.id = $document_name
WITH c ORDER BY c.index WITH collect(c) AS nodes
UNWIND range(0, size(nodes) -2) AS index
WITH nodes[index] AS start, nodes[index + 1] AS end
MERGE (start)-[:NEXT]->(end)
"""

def get_openai_client():
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )