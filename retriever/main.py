from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph_flow import build_langgraph
from models import InputState, OutputState
from fastapi.responses import RedirectResponse
import config



app = FastAPI()

# Build the LangGraph
langgraph = build_langgraph()

# Input model for API
class QuestionInput(BaseModel):
    question: str

# Output model for API
class AnswerOutput(BaseModel):
    answer: str
    analysis: str
    previous_actions: list

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/ask", response_model=AnswerOutput)
def ask_question(input_data: QuestionInput):
    try:
        result = langgraph.invoke({"question": input_data.question})
        return AnswerOutput(
            answer=result["answer"],
            analysis=result["analysis"],
            previous_actions=result.get("previous_actions", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    print(config.NEO4J_USERNAME)
    uvicorn.run(app, host="0.0.0.0", port=8001)