# 📚 Multimodal RAG Assistant – ESPRIT

Ce projet est un assistant intelligent multimodal qui répond aux requêtes utilisateurs en utilisant la **RAG (Retrieval-Augmented Generation)**, la **génération d’email automatique**, et la **recherche web intelligente**, le tout orchestré par **LangGraph**.

---

## 🗂️ Structure du projet
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

## 🧠 Fonctionnement intelligent

### 🔀 Entrée multimodale possible :
- Texte simple
- Voix (via **Moonshine STT**)
- Fichiers (supportés) :
  - PDF → `pdfplumber`, `pdf2image`
  - DOCX → `docx.Document`
  - Image → `PIL`, `pytesseract`
  - CSV, TXT

---

## 🚀 Pipeline global

1. **Prétraitement automatique**
   - Extraction de texte depuis les fichiers selon leur format
   - Création d’une base **temporaire ChromaDB** avec `sentence-transformers/paraphrase-MiniLM-L6-v2`

2. **Routage via LangGraph**
   - Si Web Search activé :
     - `web_agent` : recherche avec **DuckDuckGo**
     - Résumé multi-source avec `sshleifer/distilbart-cnn-12-6`
   - Sinon :
     - `rag_agent` :
       - Si fichier → RAG sur la base temporaire
       - Sinon → RAG sur la base persistée (`Esprit_kb/`)
     - Réponses générées via `llama3` (Ollama)

3. **📬 Génération d'Email automatique**
   - Si le prompt contient "je veux un mail" ou "génère un mail"
   - Agent email génère le message avec `llama3`
   - Affichage clair, modifiable avant envoi
   - Envoi via **SMTP**

4. **🔊 TTS automatique**
   - Lecture de la réponse via **gTTS** (si activée)

---

## 🤖 Agents LangGraph

| Agent         | Rôle                                                   |
|---------------|--------------------------------------------------------|
| `rag_agent`   | Recherche augmentée locale (KB ou fichier)             |
| `web_agent`   | Recherche Web via **DuckDuckGo** et résumé Bart        |
| `email_agent` | Génération + envoi d’emails via **llama3** et **SMTP** |

---

## ✅ Fonctionnalités clés

- Multimodalité complète (texte, voix, fichier, image)
- Recherche augmentée intelligente
- Résumé web précis basé sur actualité
- Génération & envoi d’emails utile
- Lecture vocale intégrée

---

## 🛠️ Technologies

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

## 📦 Installation

```bash
pip install -r requirements.txt
```
Configurez vos clés dans .env, puis lancez :
```bash
uvicorn app.main:app --reload
chainlit run ui/mainJdid.py
```
