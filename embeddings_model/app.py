# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os

app = FastAPI()

# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

# Load the SentenceTransformer model from a local path
model_path = "/models/USER-bge-m3"
model = SentenceTransformer(model_path)

@app.post("/predict")
async def predict(request: TextRequest):
    embeddings = model.encode([request.text], normalize_embeddings=True).tolist()
    return {"embedding": embeddings[0]}  # Return the embedding for the single input text

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
