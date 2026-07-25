import os

# Define the local model you pulled via Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Define where the vector database lives on your hard drive
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

# Define the maximum number of times the agent
#  can loop before forcing an answer
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
