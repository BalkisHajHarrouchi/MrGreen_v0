# 📚 Multimodal RAG Assistant – ESPRIT

This project is a smart **multimodal assistant** that answers user queries using **Retrieval-Augmented Generation (RAG)**, **automatic email generation**, and **intelligent web search**, all orchestrated via **LangGraph**.

---

## 📁 Project Structure

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

## 🧠 Smart Workflow

### 🔀 Supported multimodal input:
- Plain text
- Voice (via **Moonshine STT**)
- Files (supported formats):
  - PDF → `pdfplumber`, `pdf2image`
  - DOCX → `docx.Document`
  - Image → `PIL`, `pytesseract`
  - CSV, TXT

---

## 🚀 Global Pipeline

1. **Automatic Preprocessing**
   - Text extraction based on file type
   - Creation of a **temporary ChromaDB** using `sentence-transformers/paraphrase-MiniLM-L6-v2`

2. **Routing via LangGraph**
   - If Web Search is enabled:
     - `web_agent`: searches using **DuckDuckGo**
     - Multi-source summarization via `sshleifer/distilbart-cnn-12-6`
   - Otherwise:
     - `rag_agent`:
       - If a file is uploaded → RAG over temporary DB
       - Else → RAG over persistent DB (`Esprit_kb/`)
     - Answers generated using `llama3` (Ollama)

3. **📬 Automatic Email Generation**
   - Triggered if prompt contains "je veux un mail" or "génère un mail"
   - Email agent generates the message using `llama3`
   - Clear, editable preview before sending
   - Sent via **SMTP**

4. **🔊 Automatic Text-to-Speech (TTS)**
   - Answers are read aloud using **gTTS** (if enabled)

---

## 🤖 LangGraph Agents

| Agent         | Role                                                       |
|---------------|------------------------------------------------------------|
| `rag_agent`   | Local augmented search (persistent KB or uploaded file)    |
| `web_agent`   | Web search via **DuckDuckGo** + summarization (Bart)       |
| `email_agent` | Email generation + sending using **llama3** and **SMTP**   |

---

## ✅ Key Features

- Full multimodality (text, voice, files, images)
- Smart augmented retrieval
- Web summarization from live sources
- Useful email generation & delivery
- Integrated voice reading (TTS)

---

## 🛠️ Tech Stack

- LangChain + LangGraph
- Ollama (LLaMA 3)
- DuckDuckGo + Newspaper3k
- HuggingFace Transformers
- ChromaDB
- FastAPI & Chainlit
- Moonshine STT
- Google TTS (gTTS)
- SMTP (Gmail)


---

## 📦 Setup

```bash
pip install -r requirements.txt
```
Then run :
```bash
uvicorn app.main:app --reload
chainlit run ui/mainJdid.py
```
