
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from redundant_filter_retriever import RedundantFilterRetriever
# from langchain_community.llms import Ollama
# from datetime import datetime
# import langchain
# import json

# langchain.debug = True

# # --- Load the embedder ---
# EMBEDDING_MODEL = 'Lajavaness/bilingual-embedding-large'
# model_kwargs = {"trust_remote_code": True}
# embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# # --- Load the LLM ---
# model = Ollama(model="llama3")

# # --- Load existing Chroma vector store ---
# db = Chroma(
#     persist_directory="emb/db8",
#     embedding_function=embedding_function
# )

# # --- Create retriever ---
# retriever = RedundantFilterRetriever(
#     chroma=db,
#     embeddings=embedding_function
# )

# # --- Define the prompt template ---
# chat_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "Vous êtes un assistant utile. Répondez à la question suivante en utilisant uniquement le contexte fourni. "
#         "Répondez dans la même langue que la question. Si le contexte est insuffisant pour répondre, dites simplement : "
#         "'je ne sais pas'.\n"
#         "Contexte : {context}\n"
#         "Question : {question}\n"
#         "Réponse :"
#     )
# )

# # --- Create RetrievalQA chain ---
# chain = RetrievalQA.from_chain_type(
#     llm=model,
#     retriever=retriever,
#     chain_type="stuff",
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": chat_prompt},
#     verbose=True
# )

# # --- Start chatbot loop and log context ---
# log_file = "./backend/chat_log.jsonl"

# print("🧠 Chatbot is ready. Type your question below:")

# while True:
#     try:
#         question = input(">> ").strip()
#         if not question:
#             continue

#         timestamp = datetime.now().isoformat()
#         result = chain({"query": question})

#         # Extract context
#         context_used = " ".join([doc.page_content for doc in result.get("source_documents", [])]) if result.get("source_documents") else ""

#         print("\n🤖 Réponse:")
#         print(result["result"])
#         print("-" * 60)

#         # Save interaction to log file
#         log_data = {
#             "timestamp": timestamp,
#             "question": question,
#             "context": context_used,
#             "answer": result["result"]
#         }

#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

#     except KeyboardInterrupt:
#         print("\n👋 Chatbot session ended.")
#         break
#     except Exception as e:
#         print(f"⚠️ Une erreur est survenue : {e}")
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from redundant_filter_retriever import RedundantFilterRetriever
from langchain_community.llms import Ollama
from datetime import datetime
from langdetect import detect
import spacy
import langchain
import json

# Enable LangChain debug logs
langchain.debug = True

# --- Load the embedder ---
EMBEDDING_MODEL = 'Lajavaness/bilingual-embedding-large'
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# --- Load the LLM ---
model = Ollama(model="llama3")

# --- Load existing Chroma vector store ---
db = Chroma(
    persist_directory="emb/db8",
    embedding_function=embedding_function
)

# --- Create retriever ---
retriever = RedundantFilterRetriever(
    chroma=db,
    embeddings=embedding_function
)

# --- Define the prompt template ---
chat_prompt = PromptTemplate(
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

# --- Create RetrievalQA chain ---
chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": chat_prompt},
    verbose=True
)

# --- Load spaCy for tag extraction ---
try:
    nlp = spacy.load("fr_core_news_sm")
except:
    import os
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

def extract_tags(text):
    doc = nlp(text.lower())
    return list({token.lemma_ for token in doc if token.is_alpha and not token.is_stop})

# --- Start chatbot loop and log optimized data ---
log_file = "./backend/log.jsonl"

print("🧠 Chatbot is ready. Type your question below:")

while True:
    try:
        question = input(">> ").strip()
        if not question:
            continue

        timestamp = datetime.now().isoformat()
        result = chain({"query": question})

        answer = result["result"]
        language = detect(question)
        tags = extract_tags(question)

        print("\n🤖 Réponse:")
        print(answer)
        print("-" * 60)

        # Save optimized log
        log_data = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "language": language,
            "tags": tags
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        print("\n👋 Chatbot session ended.")
        break
    except Exception as e:
        print(f"⚠️ Une erreur est survenue : {e}")
