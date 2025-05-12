from fastapi import APIRouter
from app.models import RAGRequest
from app.crew.run_pipeline import run_agentic_pipeline

router = APIRouter()

@router.post("/rag")
async def rag_endpoint(req: RAGRequest):
    response = run_agentic_pipeline(req.question, use_web=False)
    return {"response": response}
