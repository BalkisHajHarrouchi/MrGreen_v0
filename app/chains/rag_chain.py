import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from app.config import CHROMA_DB_PARENT_PATH, EMBEDDING_MODEL, OLLAMA_MODEL


def get_rag_chain(query: str):
    print("🚀 Initializing RAG Chain...")
    
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )

    all_docs = []
    
    print(f"📁 Scanning Chroma vector stores under: {CHROMA_DB_PARENT_PATH}")
    for subdir in os.listdir(CHROMA_DB_PARENT_PATH):
        db_path = os.path.join(CHROMA_DB_PARENT_PATH, subdir)
        if os.path.isdir(db_path):
            try:
                print(f"🔍 Loading vector store: {subdir}")
                vs = Chroma(
                    persist_directory=db_path,
                    embedding_function=embedding_function
                )

                docs = vs.similarity_search_with_score(query, k=5)  # Now using real query

                print(f"   📦 Retrieved {len(docs)} chunks from '{subdir}'")
                for doc, score in docs:
                    doc.metadata["source"] = subdir
                    doc.metadata["score"] = score
                    all_docs.append(doc)
            except Exception as e:
                print(f"❌ Failed to load or query DB at '{db_path}': {e}")

    if not all_docs:
        raise ValueError("Aucun document pertinent trouvé dans les sous-dossiers de ChromaDB.")

    # Sort and take top 5
    all_docs = sorted(all_docs, key=lambda x: x.metadata.get("score", 1.0))[:5]

    print("\n✅ Final Top 5 Chunks Selected:")
    for i, doc in enumerate(all_docs, 1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("score", "n/a")
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i}. 📄 From '{source}' | Score: {score:.4f} | Content: {preview}...")

    # Build temporary in-memory Chroma DB
    temp_db = Chroma.from_documents(documents=all_docs, embedding=embedding_function)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Vous êtes un assistant utile. Répondez à la question suivante en utilisant uniquement le contexte fourni. "
            "Répondez dans la même langue que la question. Si le contexte est insuffisant pour répondre, dites simplement : "
            "'je ne sais pas'.\n"
            "Contexte : {context}\n"
            "Question : {question}\n"
            "Réponse :"
        )
    )


    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Vous êtes un assistant qui répond à des questions en extrayant des informations précises du contexte.\n\n"
            "📌 Utilisez uniquement les informations fournies dans le contexte.\n"
            "📌 Si la question concerne un contact, un nom ou un email, extrayez-les clairement.\n"
            "📌 Si aucune information pertinente n’est disponible, répondez : « je ne sais pas ».\n\n"
            "Contexte :\n{context}\n\n"
            "Question :\n{question}\n\n"
            "Réponse :"
        )
    )


    chain = RetrievalQA.from_chain_type(
        llm = Ollama(model=OLLAMA_MODEL, temperature=0.2, timeout=90),
        retriever=temp_db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    print("✅ RAG Chain ready.\n")
    return chain
