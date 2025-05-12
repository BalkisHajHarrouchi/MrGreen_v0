from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from app.chains.rag_chain import get_rag_chain
from app.chains.web_summary_chain import search_and_summarize_web

class AgentState(TypedDict):
    query: str
    source: Literal["rag", "web"]
    response: str

# === Agent: ChromaDB-based RAG ===
def rag_agent(state: AgentState) -> AgentState:
    print("🧠 [LangGraph] Running RAG Agent...")
    chain = get_rag_chain(state["query"])
    result_dict = chain.invoke({"query": state["query"]})
    print("✅ [RAG Agent] Response generated.")
    return {
        "query": state["query"],
        "source": "rag",
        "response": result_dict["result"]
    }

# === Agent: Web Search Summary ===
def web_agent(state: AgentState) -> AgentState:
    print("🌐 [LangGraph] Running Web Summary Agent...")
    result = search_and_summarize_web(state["query"])
    print("✅ [Web Agent] Summary generated.")
    return {
        "query": state["query"],
        "source": "web",
        "response": {
            "summary": result["summary"],
            "sources": result["sources"]
        }
    }


# === LangGraph Orchestration ===
def build_graph():
    print("🔧 [LangGraph] Building agent workflow graph...")
    graph = StateGraph(AgentState)

    # Add agent nodes
    graph.add_node("rag", rag_agent)
    graph.add_node("web", web_agent)

    # Router function that returns the name of the next node
    def route_logic(state: AgentState) -> str:
        selected = state["source"]
        print(f"🔀 [Router] Routing to agent: '{selected}'")
        return selected

    # Router node that just passes the state as-is
    def pass_through(state: AgentState) -> AgentState:
        return state

    # Add router node and routing logic
    graph.add_node("router", pass_through)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_logic, {"rag": "rag", "web": "web"})

    # End the graph after each agent
    graph.add_edge("rag", END)
    graph.add_edge("web", END)

    print("✅ [LangGraph] Workflow ready.")
    return graph.compile()
