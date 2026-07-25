from agent.graph import build_graph

def run_terminal_chat():
    print("🤖 Building Agentic RAG Graph...")
    app_workflow = build_graph()
    
    print("\n✅ System Ready! Type 'exit' or 'quit' to stop.")
    
    # We assign a static thread ID for this terminal session to enable memory
    config = {"configurable": {"thread_id": "terminal_session_1"}}
    
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down...")
            break
            
        inputs = {"question": user_input, "retries": 0}
        
        print("\n🤖 Agent:")
        # Stream the agent's thought process to the terminal
        for event in app_workflow.stream(inputs, config=config):
            for node_name, node_state in event.items():
                if node_name == "retrieve":
                    print("   [Thinking] Searching database...")
                elif node_name == "grade_documents":
                    print("   [Thinking] Grading search results...")
                elif node_name == "rewrite_query":
                    print(f"   [Action] Results irrelevant. Rewriting query to: '{node_state['question']}'")
                elif node_name == "generate":
                    print("   [Action] Generating final response...\n")
                    print(f"👉 {node_state['generation']}")

if __name__ == "__main__":
    run_terminal_chat()