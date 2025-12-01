"""
Multi-Agent System for Scientific Research Assistance
Demonstrates custom tools with ALCF Inference Endpoints
"""

from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

from clarexie_tools import (
    search_arxiv,
    lookup_molecular_properties,
    calculate_simple_statistics,
    search_protein_database
)


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    """State that tracks messages through the agent workflow."""
    messages: Annotated[list, add_messages]
    tool_call_count: int  # Track how many times we've called tools


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State) -> str:
    """
    Route to 'tools' if the last message has tool calls, otherwise route to 'synthesis'.
    Also enforce a maximum number of tool call iterations to prevent infinite loops.

    Parameters
    ----------
    state : State
        The current state containing messages and tool call count

    Returns
    -------
    str
        Either 'tools' or 'synthesis' based on whether tools need to be called
    """
    # Get tool call count, default to 0 if not present
    tool_call_count = state.get("tool_call_count", 0)
    
    # Limit tool calls to prevent infinite loops (max 10 tool iterations)
    if tool_call_count >= 10:
        print(f"⚠️  Reached maximum tool iterations ({tool_call_count}), moving to synthesis")
        return "synthesis"
    
    if isinstance(state, list):
        ai_message = state[-1]
    else:
        messages = state.get("messages", [])
        if messages:
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "synthesis"


# ============================================================
# 3. Research Agent - handles tool calls
# ============================================================
def research_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = (
        "You are a scientific research assistant with access to various databases and search tools. "
        "Your job is to gather information from multiple sources to answer user queries. "
        "Use the available tools to search for papers, look up molecular properties, search protein databases, "
        "and perform calculations as needed. "
        "IMPORTANT: After you have gathered the necessary information using tools, DO NOT call tools again. "
        "Simply provide a brief summary of what you found, and the synthesis agent will format it properly."
    ),
):
    """
    Agent that uses tools to gather scientific information.

    Parameters
    ----------
    state : State
        Current conversation state
    llm : ChatOpenAI
        Language model with tool calling capability
    tools : list
        List of available tools
    system_prompt : str
        Instructions for the agent

    Returns
    -------
    dict
        Updated state with agent's response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# ============================================================
# 4. Synthesis Agent - formats structured output
# ============================================================
def synthesis_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = (
        "You are a scientific report synthesizer. Your job is to take all the information "
        "gathered by the research agent and format it into a clear, structured summary. "
        "Organize the information logically with sections and bullet points. "
        "Highlight key findings and provide context where appropriate. "
        "Make the summary concise but comprehensive."
    ),
):
    """
    Agent that synthesizes gathered information into structured output.

    Parameters
    ----------
    state : State
        Current conversation state with all gathered information
    llm : ChatOpenAI
        Language model for synthesis
    system_prompt : str
        Instructions for synthesis

    Returns
    -------
    dict
        Updated state with synthesized response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 5. Setup LLM and tools
# ============================================================
def setup_multi_agent():
    """Initialize the multi-agent system."""
    
    # Get ALCF authentication token
    print("🔐 Authenticating with ALCF Inference Endpoints...")
    access_token = get_access_token()
    
    # Initialize the ALCF-hosted language model
    print("🤖 Initializing language model...")
    llm = ChatOpenAI(
        model_name="openai/gpt-oss-120b",
        api_key=access_token,
        base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
        temperature=0,
    )
    
    # Define available tools
    tools = [
        search_arxiv,
        lookup_molecular_properties,
        calculate_simple_statistics,
        search_protein_database
    ]
    
    return llm, tools


# ============================================================
# 6. Build the graph
# ============================================================
def increment_tool_count(state: State):
    """Increment the tool call counter."""
    current_count = state.get("tool_call_count", 0)
    return {"tool_call_count": current_count + 1}


def build_workflow_graph(llm, tools):
    """Construct the multi-agent workflow graph."""
    
    print("🏗️  Building multi-agent workflow...")
    graph_builder = StateGraph(State)
    
    # Add research agent node
    graph_builder.add_node(
        "research_agent",
        lambda state: research_agent(state, llm=llm, tools=tools),
    )
    
    # Add counter increment node
    graph_builder.add_node(
        "increment_counter",
        increment_tool_count,
    )
    
    # Add synthesis agent node
    graph_builder.add_node(
        "synthesis_agent",
        lambda state: synthesis_agent(state, llm=llm),
    )
    
    # Add tool execution node
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    
    # Define workflow edges
    # START -> research_agent
    graph_builder.add_edge(START, "research_agent")
    
    # After research_agent, route to tools or synthesis
    graph_builder.add_conditional_edges(
        "research_agent",
        route_tools,
        {"tools": "increment_counter", "synthesis": "synthesis_agent"}
    )
    
    # After incrementing counter, go to tools
    graph_builder.add_edge("increment_counter", "tools")
    
    # After tools run, go back to research_agent
    graph_builder.add_edge("tools", "research_agent")
    
    # After synthesis, end the workflow
    graph_builder.add_edge("synthesis_agent", END)
    
    # Compile the graph with increased recursion limit
    return graph_builder.compile(
        checkpointer=None,
        debug=False
    )


# ============================================================
# 7. Main execution
# ============================================================
def main():
    """Run the multi-agent system with example queries."""
    
    print("=" * 70)
    print("🔬 Scientific Research Multi-Agent System")
    print("=" * 70)
    print()
    
    # Setup
    llm, tools = setup_multi_agent()
    graph = build_workflow_graph(llm, tools)
    
    # Example queries to demonstrate different capabilities
    queries = [
        "Find recent papers about machine learning for drug discovery and tell me about the molecule aspirin",
        "Search for information about the protein hemoglobin and calculate statistics for these binding affinities: [5.2, 6.8, 4.3, 7.1, 5.9]",
        "Look up the molecular properties of caffeine and find papers about caffeine's effects on the brain",
    ]
    
    # You can uncomment this to use just one query
    # queries = [queries[0]]
    
    for idx, prompt in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"📋 Query {idx}/{len(queries)}")
        print(f"{'=' * 70}")
        print(f"❓ {prompt}")
        print(f"{'=' * 70}\n")
        
        # Stream the agent's execution with recursion limit and initial state
        for chunk in graph.stream(
            {"messages": prompt, "tool_call_count": 0},
            stream_mode="values",
            config={"recursion_limit": 50}  # Increase from default 25
        ):
            new_message = chunk["messages"][-1]
            new_message.pretty_print()
        
        print()
    
    print("=" * 70)
    print("✅ Multi-agent system execution completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
