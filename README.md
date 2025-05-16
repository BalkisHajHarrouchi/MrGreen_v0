# 🎓 Multimodal RAG Assistant – Esprit School of Engineering

This project was developed as part of the coursework at **Esprit School of Engineering**, aiming to explore advanced AI applications through **Retrieval-Augmented Generation (RAG)**, **automatic email generation**, and **intelligent web search**, orchestrated using **LangGraph**.

---

## 🔍 Overview

This assistant supports **multimodal input** (text, voice, files) and intelligently routes requests using dedicated agents to answer questions, summarize the web, or generate emails. It demonstrates the integration of **FastAPI**, **Chainlit**, **ChromaDB**, and **LangChain** for smart academic assistants.

---

## 🚀 Features

- Retrieval-Augmented Generation (RAG) with local and temporary knowledge bases
- Multimodal input: text, voice (Moonshine), and documents (PDF, DOCX, images)
- Live web summarization using DuckDuckGo and Bart
- Automatic email generation with customizable previews
- Text-to-speech output using gTTS

---

## 🧠 Tech Stack

### Frontend
- **Chainlit** (custom interface with TTS and file upload)

### Backend
- **FastAPI**
- **LangChain**, **LangGraph**, **Ollama (LLaMA 3)**

### Tools & Services
- **ChromaDB**, **HuggingFace Transformers**
- **Moonshine STT**, **gTTS (Google TTS)**
- **SMTP (Gmail)** for email delivery

---

## 📁 Directory Structure


```text
project/
├── .env                        # Environment variables (e.g., GROQ_API_KEY)
├── requirements.txt           # Python dependencies (langchain, fastapi, etc.)

├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Global configuration
│
│   ├── api/
│   │   ├── rag.py             # POST endpoint /rag (local knowledge base)
│   │   └── web_summary_api.py # POST endpoint /websummary (web summarization)
│
│   ├── chains/
│   │   ├── file_chain.py          # Ingest uploaded file into temporary Chroma DB
│   │   ├── rag_chain.py           # RAG over persisted Chroma vector DBs
│   │   └── web_summary_chain.py   # Summarize articles from web using DuckDuckGo
│
│   ├── agents/
│   │   └── langgraph_workflow.py  # LangGraph agent graph: RAG, Web, and Email
│
│   └── models.py              # Pydantic request/response schemas

├── Esprit_kb/                 # Persisted ChromaDB knowledge base
│   └── ESE/
│       ├── kb_contacts/
│       ├── kb_plan_etude_Telecom/
│       └── kb_plan_etude_genie_civil/

├── ui/
│   └── mainJdid.py            # Chainlit UI (voice, file upload, TTS toggle)
```

---

## 🌐 Workflow (LangGraph Agents)

| Agent         | Role                                                             |
|---------------|------------------------------------------------------------------|
| `rag_agent`   | Retrieves from persistent or temporary ChromaDBs                 |
| `web_agent`   | Web search via DuckDuckGo, summarized with `distilbart-cnn-12-6` |
| `email_agent` | Generates/send emails via `llama3` and Gmail SMTP               |



---


## 📦 Getting started

```bash
pip install -r requirements.txt
```
Then run :
```bash
uvicorn app.main:app --reload
chainlit run ui/mainJdid.py
```

## Acknowledgments

This project was carried out as part of an academic initiative by **Esprit School of Engineering**, under the supervision of faculty advisors. It highlights the practical integration of AI tools within modern education.

