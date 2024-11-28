import os

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://0.0.0.0:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "<neo4j_user>")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "<password>")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "<neo4j_database>")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8004/v1")
BOS_TOKEN = '<|begin_of_text|>'
EOS_TOKEN = '<|end_of_text|>'
MODEL_NAME = "vikhr_llama"
TEMPERATURE = 0.01
MAX_TOKENS = 1000

# Embeddings server URL
EMBEDDINGS_SERVER_URL = os.getenv("EMBEDDINGS_SERVER_URL", "http://0.0.0.0:8000")