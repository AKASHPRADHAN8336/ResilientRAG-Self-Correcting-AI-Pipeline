from langchain_core.output_parsers import StrOutputParser
from models.ollama_model import llm
from prompts.prompt_templates import GRADER_PROMPT, REWRITER_PROMPT, GENERATOR_PROMPT
from rag.vector_store import get_retriever

retriever = get_retriever()

def retrieve(state):
    question = state["question"]
    docs = retriever.invoke(question)
    doc_texts = [d.page_content for d in docs]
    return {"documents": doc_texts, "question": question, "retries": state.get("retries", 0)}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    
    grader_chain = GRADER_PROMPT | llm | StrOutputParser()
    
    filtered_docs = []
    for doc in documents:
        score = grader_chain.invoke({"question": question, "document": doc})
        if "yes" in score.lower():
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs, "question": question, "retries": state["retries"]}

def rewrite_query(state):
    question = state["question"]
    rewrite_chain = REWRITER_PROMPT | llm | StrOutputParser()
    better_question = rewrite_chain.invoke({"question": question})
    
    return {"question": better_question, "documents": state["documents"], "retries": state["retries"] + 1}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(documents)
    
    rag_chain = GENERATOR_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": answer, "question": question, "documents": documents, "retries": state["retries"]}
