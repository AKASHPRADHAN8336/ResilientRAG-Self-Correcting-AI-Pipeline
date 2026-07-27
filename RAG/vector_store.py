from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_PATH, COLLECTION_NAME

# Shared embeddings model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_vector_store():
    """Returns the ChromaDB connection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def get_retriever():
    """Returns the retriever interface for LangGraph."""
    db = get_vector_store()
    return db.as_retriever(search_kwargs={"k": 3})
