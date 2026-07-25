import streamlit as st
from langgraph.graph import build_graph

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Agentic RAG | Local AI",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# --- CACHE THE GRAPH ---
# We use st.cache_resource so the graph only compiles once when the app starts


@st.cache_resource
def get_workflow():
    return build_graph()


app_workflow = get_workflow()


with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    **Self-Correcting Agentic RAG**

    This application demonstrates an advanced
    Retrieval-Augmented Generation (RAG) pipeline.
    Unlike standard RAG, this system uses an
    **Agentic Workflow** to evaluate its own retrieved context.

    If the database returns irrelevant data,
    the agent autonomously rewrites the search
    query and searches again, preventing hallucinations.

    **Tech Stack:**
    * **LLM:** Llama 3 (Local via Ollama)
    * **Orchestration:** LangGraph
    * **Vector DB:** ChromaDB
    * **Embeddings:** HuggingFace `BGE-small`
    """)

# --- MAIN UI ---
st.title("🕵️‍♂️ Agentic RAG Chatbot")
st.markdown(
    "Ask a question based on the ingested documents. "
    "Watch the agent evaluate its own search results in real-time."
    )

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about the policy..."):
    
    # Add user message to chat UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent's response area
    with st.chat_message("assistant"):
        # We use a status container to show the agent's "thoughts"
        with st.status("Agent is thinking...", expanded=True) as status_box:

            # Run the LangGraph workflow using .stream()
            # This yields the state after EVERY
            #  node executes so we can track it
            inputs = {"question": user_input, "retries": 0}
            final_generation = ""

            for event in app_workflow.stream(inputs):
                for node_name, node_state in event.items():

                    if node_name == "retrieve":
                        st.write(
                            "🔍 **Retrieving documents"
                            "** from vector database..."
                        )

                    elif node_name == "grade_documents":
                        st.write(
                            "⚖️ **Grading documents** for " 
                            "relevance to the question..."
                        )

                    elif node_name == "rewrite_query":
                        # If we hit this node, it means the
                        # grader failed the documents!
                        st.write(
                            "⚠️ **Documents irrelevant."
                            "** Rewriting search query..."
                        )
                        st.write(f"👉 *New Query: {node_state['question']}*")

                    elif node_name == "generate":
                        st.write(
                            "✅ **Relevant context found."
                            "** Generating final response..."
                        )
                        final_generation = node_state["generation"]

            status_box.update(
                label="Complete!", state="complete", expanded=False
            )

        # Display the final answer outside the status box
        st.markdown(final_generation)

    # Save assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": final_generation}
    )
