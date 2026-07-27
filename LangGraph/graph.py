from langgraph.graph import StateGraph, START, END
from config import MAX_RETRIES
from langgraph.state import GraphState
from langgraph.nodes import retrieve, grade_documents, rewrite_query, generate

def decide_to_generate(state):
    filtered_documents = state["documents"]
    retries = state["retries"]
    
    if retries >= MAX_RETRIES:
        return "generate"
    if len(filtered_documents) == 0:
        return "rewrite"
    return "generate"

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents", decide_to_generate,
        {"rewrite": "rewrite_query", "generate": "generate"}
    )
    
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
