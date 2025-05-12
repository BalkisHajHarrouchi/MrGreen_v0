project/
│
├── .env                            # Variables d’environnement (ex: GROQ_API_KEY)
├── requirements.txt               # Dépendances Python (langchain, langgraph, fastapi, etc.)
│
├── app/
│   ├── main.py                    # Entrypoint FastAPI
│   ├── config.py                  # Configuration globale (modèles, chemins, etc.)
│
│   ├── api/
│   │   └── rag.py                 # Endpoint POST `/rag` (KB locale)
│   │   └── web_summary_api.py     # Endpoint POST `/websummary` (résumé web)
│
│   ├── chains/
│   │   └── file_chain.py          # Ingestion d’un fichier dans une base temporaire
│   │   └── rag_chain.py           # Recherche RAG dans les bases persistées
│   │   └── web_summary_chain.py   # Résumé multi-source depuis le web
│
│   ├── agents/
│   │   └── langgraph_workflow.py  # Graphe LangGraph : routing RAG / Web Search
│
│   └── models.py                  # Schemas Pydantic des requêtes
│
├── Esprit_kb/                     # Base ChromaDB persistée
│   └── ESE/
│       └── kb_contacts/
│       └── kb_plan_etude_Telecom/
│       └── kb_plan_etude_genie_civil/
│
├── ui/
│   └── mainJdid.py                # Interface Chainlit (voix, TTS, Web Search toggle)
