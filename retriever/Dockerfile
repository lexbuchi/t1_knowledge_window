# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the application port
EXPOSE 8001

# Set the environment variables (optional defaults can be overridden)
ENV NEO4J_URI="bolt://0.0.0.0:7687" \
    NEO4J_USERNAME="<neo4j_user>" \
    NEO4J_PASSWORD="<password>" \
    NEO4J_DATABASE="<neo4j_database>" \
    OPENAI_API_KEY="EMPTY" \
    OPENAI_API_BASE="http://0.0.0.0:8004/v1" \
    EMBEDDINGS_SERVER_URL="http://0.0.0.0:8000"

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]