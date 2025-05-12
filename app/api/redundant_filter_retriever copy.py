from fastapi import APIRouter
from pydantic import BaseModel
from app.chains.web_summary import search_and_summarize_web

router = APIRouter()

class WebSummaryRequest(BaseModel):
    query: str
    max_results: int = 3

@router.post("/websummary")
async def web_summary(request: WebSummaryRequest):
    result = search_and_summarize_web(request.query, request.max_results)
    return result
