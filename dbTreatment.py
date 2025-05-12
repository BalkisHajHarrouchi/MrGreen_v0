from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from redundant_filter_retriever import RedundantFilterRetriever
import langchain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama

langchain.debug = True

# Loading the embedder
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# Define text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n\n"
)

# Load and split the document
loader = TextLoader("kb_contacts.txt", encoding="utf-8")
docs = loader.load_and_split(text_splitter=text_splitter)

# Initialize the vector store
db = Chroma(
    persist_directory="Esprit_KB/ESE/kb_contacts",
    embedding_function=embedding_function
)

# Add documents to the vector store
db.add_texts(texts=[doc.page_content for doc in docs])

# Create the retriever
retriever = RedundantFilterRetriever(
    chroma=db,
    embeddings=embedding_function
)
