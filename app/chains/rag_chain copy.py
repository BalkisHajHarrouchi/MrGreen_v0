from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.api.redundant_filter_retriever import RedundantFilterRetriever
from langchain_community.llms import Ollama
import langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

from app.config import EMBEDDING_MODEL, CHROMA_DB_PATH, OLLAMA_MODEL

langchain.debug = True

def get_rag_chain():
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )

    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_function
    )

    retriever = RedundantFilterRetriever(
        chroma=db,
        embeddings=embedding_function
    )

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

    chain = RetrievalQA.from_chain_type(
        llm=Ollama(model=OLLAMA_MODEL),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    return chain
