from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
    num_predict=40
)

response = llm.invoke("what is science")
print(response.content)
print("Response type:", type(response))
