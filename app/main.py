from fastapi import FastAPI
from app.api import rag, web_summary_api

app = FastAPI(
    title="Agentic RAG API",
    description="FastAPI app powered by CrewAI agents for document RAG and web summarization.",
    version="1.0.0"
)

# Include API routes
app.include_router(rag.router)
app.include_router(web_summary_api.router)
