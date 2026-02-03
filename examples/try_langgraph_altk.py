"""
Try ALTK LangGraph Integration

Run this example:
    cd /Users/anu/agent-lifecycle-toolkit
    python examples/try_langgraph_altk.py

Requirements:
    pip install langgraph langchain-core
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Annotated, TypedDict, Optional
import operator

# Check for langgraph
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Please install langgraph: pip install langgraph langchain-core")
    sys.exit(1)

# ALTK imports
from altk.integrations.langgraph import (
    create_validation_node,
    create_review_node,
    ValidationDecision,
    ReviewOutcome,
    ALTKEnhancedGraph,
    ALTKGraphConfig,
)


# =============================================================================
# Define Tool Specifications
# =============================================================================

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g., 'San Francisco')",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the product database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# =============================================================================
# Define State
# =============================================================================

class AgentState(TypedDict, total=False):
    """State for our ALTK-enhanced agent."""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    tool_specs: List[Dict[str, Any]]
    current_tool_call: Optional[Dict[str, Any]]
    tool_response: Optional[Any]
    validation_result: Optional[Any]
    review_result: Optional[Any]
    iteration: int


# =============================================================================
# Define Nodes
# =============================================================================

def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Simulated agent that decides to call a tool.
    In a real app, this would be an LLM call.
    """
    iteration = state.get("iteration", 0)

    # Check if we got a validation rejection
    validation_result = state.get("validation_result")
    if validation_result and validation_result.decision == ValidationDecision.REJECT:
        print(f"  [Agent] Validation rejected! Issues: {[i.explanation for i in validation_result.issues]}")
        return {
            "messages": [{
                "role": "assistant",
                "content": f"I need to fix my tool call. Validation issues found.",
            }],
            "iteration": iteration + 1,
        }

    # Check if we got a bad review
    review_result = state.get("review_result")
    if review_result and review_result.outcome == ReviewOutcome.NOT_ACCOMPLISHED:
        print(f"  [Agent] Review detected issues: {review_result.details}")
        return {
            "messages": [{
                "role": "assistant",
                "content": "The tool response had issues, let me try again.",
            }],
            "iteration": iteration + 1,
        }

    # Prevent infinite loops
    if iteration > 2:
        return {
            "messages": [{"role": "assistant", "content": "Task complete."}],
            "current_tool_call": None,
        }

    # Simulate deciding to call get_weather
    print(f"  [Agent] Deciding to call get_weather tool...")
    return {
        "messages": [{
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [{
                "id": f"call_{iteration}",
                "name": "get_weather",
                "arguments": {"city": "San Francisco", "units": "celsius"},
            }],
        }],
        "current_tool_call": {
            "name": "get_weather",
            "arguments": {"city": "San Francisco", "units": "celsius"},
        },
        "iteration": iteration + 1,
    }


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Simulated tool execution.
    In a real app, this would actually call the tool.
    """
    tool_call = state.get("current_tool_call", {})
    tool_name = tool_call.get("name", "unknown")
    args = tool_call.get("arguments", {})

    print(f"  [Tool] Executing {tool_name} with args: {args}")

    # Simulate tool response
    if tool_name == "get_weather":
        response = {
            "temperature": 18,
            "units": args.get("units", "celsius"),
            "conditions": "partly cloudy",
            "humidity": 65,
            "city": args.get("city", "unknown"),
        }
    elif tool_name == "search_database":
        response = {
            "results": [
                {"id": 1, "name": "Product A", "price": 29.99},
                {"id": 2, "name": "Product B", "price": 49.99},
            ],
            "total": 2,
            "query": args.get("query", ""),
        }
    else:
        response = {"error": f"Unknown tool: {tool_name}"}

    print(f"  [Tool] Response: {response}")

    return {
        "tool_response": response,
        "messages": [{
            "role": "tool",
            "tool_call_id": f"call_{state.get('iteration', 0)}",
            "content": str(response),
        }],
    }


# =============================================================================
# Routing Functions
# =============================================================================

def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    if state.get("current_tool_call") is None:
        return "end"
    return "validate"


def route_after_validation(state: AgentState) -> str:
    """Route based on validation result."""
    validation_result = state.get("validation_result")

    if validation_result is None:
        print("  [Router] No validation result, proceeding to tools")
        return "tools"

    print(f"  [Router] Validation decision: {validation_result.decision}")

    if validation_result.decision == ValidationDecision.APPROVE:
        return "tools"
    else:
        # Reject or Error - go back to agent
        return "agent"


def route_after_review(state: AgentState) -> str:
    """Route based on review result."""
    review_result = state.get("review_result")

    if review_result is None:
        return "agent"

    print(f"  [Router] Review outcome: {review_result.outcome}")

    # Always return to agent (it will decide what to do)
    return "agent"


# =============================================================================
# Build and Run the Workflow
# =============================================================================

def main():
    print("=" * 60)
    print("ALTK LangGraph Integration Demo")
    print("=" * 60)

    # Create LLM client for ALTK components
    print("\n1. Creating ALTK nodes...")

    from altk.core.llm import get_llm
    LLMClass = get_llm("openai.sync.output_val")
    llm_client = LLMClass(model="gpt-4o-mini")
    print("   - LLM client created (openai.sync.output_val / gpt-4o-mini)")

    validation_node = create_validation_node(
        tool_specs=TOOL_SPECS,
        track="fast_track",
        use_sparc=True,
        use_refraction=False,
        llm_client=llm_client,
    )
    print("   - ValidationNode created (fast_track)")

    review_node = create_review_node(review_type="json", llm_client=llm_client)
    print("   - ReviewNode created (JSON review)")

    # Build the workflow
    print("\n2. Building LangGraph workflow...")

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("review", review_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"validate": "validate", "end": END}
    )

    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"tools": "tools", "agent": "agent"}
    )

    workflow.add_edge("tools", "review")

    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {"agent": "agent"}
    )

    # Compile
    app = workflow.compile()
    print("   - Workflow compiled successfully!")

    # Run the workflow
    print("\n3. Running workflow...")
    print("-" * 60)

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "tool_specs": TOOL_SPECS,
        "iteration": 0,
    }

    try:
        result = app.invoke(initial_state)

        print("-" * 60)
        print("\n4. Final Results:")
        print(f"   - Total messages: {len(result.get('messages', []))}")
        print(f"   - Iterations: {result.get('iteration', 0)}")

        validation_result = result.get("validation_result")
        if validation_result:
            print(f"   - Validation: {validation_result.decision}")
            if validation_result.issues:
                print(f"     Issues: {[i.explanation for i in validation_result.issues]}")

        review_result = result.get("review_result")
        if review_result:
            print(f"   - Review: {review_result.outcome}")

        tool_response = result.get("tool_response")
        if tool_response:
            print(f"   - Tool response: {tool_response}")

    except Exception as e:
        print(f"\nError running workflow: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


# =============================================================================
# Alternative: Using ALTKEnhancedGraph wrapper
# =============================================================================

def main_with_wrapper():
    """Alternative demo using ALTKEnhancedGraph wrapper."""
    print("\n" + "=" * 60)
    print("ALTK Enhanced Graph Wrapper Demo")
    print("=" * 60)

    # Create base workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Create config
    config = ALTKGraphConfig(
        validation_enabled=True,
        review_enabled=True,
        repair_enabled=False,
        tool_specs=TOOL_SPECS,
        validation_track="syntax",
        on_validation_fail="warn",  # Warn but continue
    )

    # Enhance the graph
    enhanced = ALTKEnhancedGraph(
        graph=workflow,
        config=config,
        tool_node_name="tools",
        agent_node_name="agent",
    )

    # Add ALTK nodes
    enhanced.enhance()

    # Wire up edges manually for this demo
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "altk_validate")
    workflow.add_edge("altk_validate", "tools")
    workflow.add_edge("tools", "altk_review")
    workflow.add_edge("altk_review", END)

    app = workflow.compile()

    print("\nRunning enhanced workflow...")
    result = app.invoke({
        "messages": [{"role": "user", "content": "Weather please!"}],
        "tool_specs": TOOL_SPECS,
        "iteration": 0,
        "current_tool_call": {"name": "get_weather", "arguments": {"city": "NYC"}},
    })

    print(f"\nResult: {result.get('validation_result')}")
    print(f"Review: {result.get('review_result')}")


if __name__ == "__main__":
    main()

    # Uncomment to try the wrapper version:
    # main_with_wrapper()
