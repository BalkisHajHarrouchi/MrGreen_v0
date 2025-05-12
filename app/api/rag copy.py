from fastapi import APIRouter
from app.models import RAGRequest
from app.chains.rag_chain import get_rag_chain

router = APIRouter()
chain = get_rag_chain()

@router.post("/rag")
async def rag_endpoint(req: RAGRequest):
    print(f"📥 Received question: {req.question!r}")
    result = chain({"query": req.question})
    return {"response": result["result"]}

