from fastapi import APIRouter
from pydantic import BaseModel
from app.crew.run_pipeline import run_agentic_pipeline

router = APIRouter()

class WebSummaryRequest(BaseModel):
    query: str

@router.post("/websummary")
async def web_summary(request: WebSummaryRequest):
    response = run_agentic_pipeline(request.query, use_web=True)
    return {"summary": response}
