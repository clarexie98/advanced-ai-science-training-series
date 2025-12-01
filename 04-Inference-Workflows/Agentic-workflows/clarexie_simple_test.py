"""
Simplified version of the multi-agent system for testing and debugging.
Uses only ONE simple query to verify the system works.
"""

from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

from clarexie_tools import (
    lookup_molecular_properties,
    calculate_simple_statistics,
)


# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


# Routing logic with iteration limit
def route_tools(state: State) -> str:
    tool_call_count = state.get("tool_call_count", 0)
    
    if tool_call_count >= 5:
        print(f"⚠️  Reached maximum tool iterations ({tool_call_count}), moving to synthesis")
        return "synthesis"
    
    if isinstance(state, list):
        ai_message = state[-1]
    else:
        messages = state.get("messages", [])
        if messages:
            ai_message = messages[-1]
        else:
            return "synthesis"
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "synthesis"


# Research agent
def research_agent(state: State, llm: ChatOpenAI, tools: list):
    system_prompt = (
        "You are a scientific assistant. Use tools to gather information. "
        "After calling tools ONCE and getting results, provide a brief summary. "
        "DO NOT call the same tool multiple times."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# Synthesis agent
def synthesis_agent(state: State, llm: ChatOpenAI):
    system_prompt = "Provide a clear, concise summary of the information gathered."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    result = llm.invoke(messages)
    return {"messages": [result]}


# Increment counter
def increment_tool_count(state: State):
    current_count = state.get("tool_call_count", 0)
    return {"tool_call_count": current_count + 1}


def main():
    print("=" * 70)
    print("🔬 SIMPLIFIED Multi-Agent Test")
    print("=" * 70)
    print()
    
    # Setup
    print("🔐 Authenticating...")
    access_token = get_access_token()
    
    print("🤖 Initializing LLM...")
    llm = ChatOpenAI(
        model_name="openai/gpt-oss-120b",
        api_key=access_token,
        base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
        temperature=0,
    )
    
    # Use only 2 simple tools
    tools = [lookup_molecular_properties, calculate_simple_statistics]
    
    print("🏗️  Building workflow...")
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("research_agent", lambda state: research_agent(state, llm, tools))
    graph_builder.add_node("synthesis_agent", lambda state: synthesis_agent(state, llm))
    graph_builder.add_node("increment_counter", increment_tool_count)
    
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_edge(START, "research_agent")
    graph_builder.add_conditional_edges(
        "research_agent",
        route_tools,
        {"tools": "increment_counter", "synthesis": "synthesis_agent"}
    )
    graph_builder.add_edge("increment_counter", "tools")
    graph_builder.add_edge("tools", "research_agent")
    graph_builder.add_edge("synthesis_agent", END)
    
    graph = graph_builder.compile()
    
    # SIMPLE TEST QUERY - only needs 1 tool call
    print("\n" + "=" * 70)
    print("📋 Test Query")
    print("=" * 70)
    
    prompt = "Look up the molecular properties of water"
    print(f"❓ {prompt}")
    print("=" * 70 + "\n")
    
    try:
        for chunk in graph.stream(
            {"messages": prompt, "tool_call_count": 0},
            stream_mode="values",
            config={"recursion_limit": 50}
        ):
            new_message = chunk["messages"][-1]
            new_message.pretty_print()
        
        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see recursion errors, the LLM may be calling tools repeatedly.")
        print("Try simplifying the query or adjusting the system prompt.")


if __name__ == "__main__":
    main()
