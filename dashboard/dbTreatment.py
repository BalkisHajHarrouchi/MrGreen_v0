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
EMBEDDING_MODEL = 'Lajavaness/bilingual-embedding-large'

model_kwargs = {"trust_remote_code": True}
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

# Define text splitter with chunk size and overlap
text_splitter = CharacterTextSplitter(
    chunk_size=300,  # Desired chunk size
    chunk_overlap=50,  # No overlap between chunks
    #length_function=len,  # Use length based on character count
    separator="\n\n"
)

# Load and split the document

loader = TextLoader("combination.txt", encoding="utf-8")
docs = loader.load_and_split(text_splitter=text_splitter)


# Initialize the vector store
db = Chroma(
    persist_directory="emb/db8",
    embedding_function=embedding_function
)

# Add documents to the vector store
db.add_texts(texts=[doc.page_content for doc in docs])


# Create the retriever
retriever = RedundantFilterRetriever(
    chroma=db, 
    embeddings=embedding_function
)



