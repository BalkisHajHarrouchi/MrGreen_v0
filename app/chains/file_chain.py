import os
import shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from app.chains.redundant_filter_retriever import RedundantFilterRetriever
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Model config
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True}
)



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ⬆ more context per chunk
    chunk_overlap=100,    # ⬆ ensures things like emails aren’t split
    separators=["\n\n", "\n", ".", " "]  # ⬅️ allows better boundary control
)


# === Build chain from a temp .txt source
def create_chain_from_text_file(txt_path: str, persist_dir: str):
    # Load and split text
    docs = TextLoader(txt_path, encoding="utf-8").load()
    chunks = text_splitter.split_documents(docs)

    # Ensure directory is empty
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # Build vector store from chunks
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    db.persist()

    # Build retriever
    retriever = RedundantFilterRetriever(
        chroma=db,
        embeddings=embedding_function
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Vous êtes un assistant qui répond uniquement en se basant sur le contexte fourni ci-dessous.\n\n"
            "- Utilisez des informations spécifiques telles que noms, emails, adresses si elles sont présentes.\n"
            "- Si vous voyez un formulaire ou une adresse email utile, incluez-les dans la réponse.\n"
            "- Si aucune information pertinente n’est trouvée, répondez simplement : 'je ne sais pas'.\n\n"
            "Contexte :\n{context}\n\n"
            "Question :\n{question}\n\n"
            "Réponse :"
        )
    )


    # Build RetrievalQA chain
    llm = Ollama(model="llama3", temperature=0.2)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, persist_dir

