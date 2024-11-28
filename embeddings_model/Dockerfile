# Base image with PyTorch and necessary libraries
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set up environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install transformers and other Python dependencies
COPY requirements.txt .

# RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files into the container
WORKDIR /app
COPY app.py /app
COPY USER-bge-m3 /models/USER-bge-m3
# Download the model using Hugging Face
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('intfloat/multilingual-e5-large')"

# Run a basic server to host the model (e.g., FastAPI)
CMD ["python3", "app.py"]
