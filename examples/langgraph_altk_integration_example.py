"""Example: ALTK Integration with LangGraph

This example demonstrates how to use ALTK components with LangGraph
to create an agent with built-in validation, review, and repair capabilities.

The example shows three different approaches:
1. Manual node wiring - Full control over the workflow
2. ALTKEnhancedGraph wrapper - Quick enhancement of existing graphs
3. create_altk_workflow - One-liner for standard agent patterns

To enable SPARC/review/repair functionality, set environment variables:
    export ALTK_LLM_PROVIDER=litellm.ollama.output_val
    export ALTK_MODEL_NAME=granite3.3

Note: Use the `.output_val` variant for SPARC semantic validation.
"""

import os
from typing import Dict, Any, List, Annotated, TypedDict, Optional
import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
except ImportError:
    print("This example requires langgraph and langchain-core.")
    print("Install with: pip install langgraph langchain-core")
    exit(1)

# ALTK imports
from altk.integrations.langgraph import (
    # Types
    ALTKState,
    ValidationDecision,
    ReviewOutcome,
    ToolCallInfo,
    # Node factories
    create_validation_node,
    create_review_node,
    create_repair_node,
    # Convenience wrappers
    ALTKEnhancedGraph,
    ALTKGraphConfig,
    create_altk_workflow,
)
from altk.core.llm import get_llm


# =============================================================================
# LLM Client Setup (optional, for review/repair functionality)
# =============================================================================

def get_llm_client() -> Optional[Any]:
    """Get LLM client from environment variables if configured."""
    try:
        client = get_llm("auto_from_env")()
        if client._chosen_provider is not None:
            return client
    except Exception:
        pass
    return None


# Global LLM client (None if not configured)
LLM_CLIENT = get_llm_client()


# =============================================================================
# Example Tool Specifications
# =============================================================================

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# =============================================================================
# Example 1: Manual Node Wiring
# =============================================================================


def example_manual_wiring():
    """Manually wire ALTK nodes into a LangGraph workflow.

    This gives you full control over the workflow structure.
    """
    print("\n" + "=" * 60)
    print("Example 1: Manual Node Wiring")
    print("=" * 60)

    # Define state
    class AgentState(TypedDict, total=False):
        messages: Annotated[List[Dict[str, Any]], operator.add]
        tool_specs: List[Dict[str, Any]]
        current_tool_call: Dict[str, Any]
        tool_response: Any
        validation_result: Any
        review_result: Any

    # Create nodes
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Simulated agent that decides to call a tool."""
        messages = state.get("messages", [])

        # Check if we got a validation rejection
        validation_result = state.get("validation_result")
        if validation_result and validation_result.decision == ValidationDecision.REJECT:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"I cannot proceed with this tool call. Issues: {validation_result.issues}",
                    }
                ]
            }

        # Simulate deciding to call a tool
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "name": "get_weather",
                            "arguments": {"city": "San Francisco", "units": "celsius"},
                        }
                    ],
                }
            ],
            "current_tool_call": {
                "name": "get_weather",
                "arguments": {"city": "San Francisco", "units": "celsius"},
            },
        }

    def tool_node(state: AgentState) -> Dict[str, Any]:
        """Simulated tool execution."""
        tool_call = state.get("current_tool_call", {})

        # Simulate tool response
        response = {
            "temperature": 18,
            "units": "celsius",
            "conditions": "partly cloudy",
            "city": tool_call.get("arguments", {}).get("city", "unknown"),
        }

        return {
            "tool_response": response,
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": str(response),
                }
            ],
        }

    # Create ALTK nodes
    validation_node = create_validation_node(
        tool_specs=TOOL_SPECS,
        track="fast_track",
        llm_client=LLM_CLIENT,  # Enable SPARC if LLM configured
    )

    # Review node requires an LLM client
    review_node = create_review_node(
        llm_client=LLM_CLIENT,
        review_type="json",
    ) if LLM_CLIENT else None

    # Build the workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("tools", tool_node)
    if review_node:
        workflow.add_node("review", review_node)

    # Wire up edges
    workflow.set_entry_point("agent")

    # Agent -> Validate
    workflow.add_edge("agent", "validate")

    # Validate -> Tools or back to Agent
    def route_after_validation(state: AgentState) -> str:
        validation_result = state.get("validation_result")
        if validation_result and validation_result.decision == ValidationDecision.APPROVE:
            return "tools"
        return "agent"

    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"tools": "tools", "agent": "agent"},
    )

    # Tools -> Review (if available) -> End
    if review_node:
        workflow.add_edge("tools", "review")
        workflow.add_edge("review", END)
    else:
        workflow.add_edge("tools", END)

    # Compile
    app = workflow.compile()

    # Run
    initial_state = {
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "tool_specs": TOOL_SPECS,
    }

    print("\nRunning workflow...")
    result = app.invoke(initial_state)

    print("\nFinal state:")
    print(f"  Messages: {len(result.get('messages', []))} messages")
    print(f"  Validation: {result.get('validation_result')}")
    print(f"  Review: {result.get('review_result')}")

    return result


# =============================================================================
# Example 2: Using ALTKEnhancedGraph Wrapper
# =============================================================================


def example_enhanced_graph():
    """Use ALTKEnhancedGraph to quickly enhance an existing workflow."""
    print("\n" + "=" * 60)
    print("Example 2: ALTKEnhancedGraph Wrapper")
    print("=" * 60)

    # Define state
    class AgentState(TypedDict, total=False):
        messages: Annotated[List[Dict[str, Any]], operator.add]
        tool_specs: List[Dict[str, Any]]
        current_tool_call: Dict[str, Any]
        tool_response: Any
        validation_result: Any
        review_result: Any
        repair_result: Any
        altk_metadata: Dict[str, Any]

    # Simple agent and tool nodes
    def agent_node(state: AgentState) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Searching for products...",
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "name": "search_products",
                            "arguments": {"query": "laptop", "max_results": 10},
                        }
                    ],
                }
            ],
            "current_tool_call": {
                "name": "search_products",
                "arguments": {"query": "laptop", "max_results": 10},
            },
        }

    def tool_node(state: AgentState) -> Dict[str, Any]:
        return {
            "tool_response": {
                "results": [
                    {"name": "MacBook Pro", "price": 1999},
                    {"name": "Dell XPS", "price": 1499},
                ],
                "total": 2,
            }
        }

    # Create base workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Enhance with ALTK
    config = ALTKGraphConfig(
        validation_enabled=True,
        review_enabled=LLM_CLIENT is not None,  # Only enable if LLM configured
        repair_enabled=False,
        tool_specs=TOOL_SPECS,
        validation_track="fast_track",
        on_validation_fail="warn",  # Warn but continue
        llm_client=LLM_CLIENT,
    )

    enhanced = ALTKEnhancedGraph(
        graph=workflow,
        config=config,
        tool_node_name="tools",
        agent_node_name="agent",
    )

    # The enhance() method adds ALTK nodes
    enhanced.enhance()

    # Manual edge wiring for this example
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "altk_validate")
    workflow.add_edge("altk_validate", "tools")
    if LLM_CLIENT:
        workflow.add_edge("tools", "altk_review")
        workflow.add_edge("altk_review", END)
    else:
        workflow.add_edge("tools", END)

    app = workflow.compile()

    # Run
    initial_state = {
        "messages": [{"role": "user", "content": "Find me a laptop"}],
        "tool_specs": TOOL_SPECS,
    }

    print("\nRunning enhanced workflow...")
    result = app.invoke(initial_state)

    print("\nFinal state:")
    print(f"  Validation result: {result.get('validation_result')}")
    print(f"  Review result: {result.get('review_result')}")

    return result


# =============================================================================
# Example 3: One-liner with create_altk_workflow
# =============================================================================


def example_one_liner():
    """Use create_altk_workflow for the simplest setup."""
    print("\n" + "=" * 60)
    print("Example 3: One-liner with create_altk_workflow")
    print("=" * 60)

    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple agent that calls get_weather once, then returns final response."""
        # Check if we already have a tool response - if so, return final answer
        if state.get("tool_response"):
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"The weather is: {state['tool_response']}",
                    }
                ],
                "current_tool_call": None,  # No more tool calls
            }

        # First call - request weather
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Checking weather...",
                    "tool_calls": [
                        {
                            "id": "call_789",
                            "name": "get_weather",
                            "arguments": {"city": "New York"},
                        }
                    ],
                }
            ],
            "current_tool_call": {
                "name": "get_weather",
                "arguments": {"city": "New York"},
            },
        }

    def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple tool executor."""
        return {
            "tool_response": {"temperature": 22, "conditions": "sunny"},
        }

    # One line to create a fully ALTK-enhanced workflow
    app = create_altk_workflow(
        agent_node=agent_node,
        tool_node=tool_node,
        tool_specs=TOOL_SPECS,
        validation_track="fast_track",
        review_enabled=LLM_CLIENT is not None,
        altk_llm_client=LLM_CLIENT,
    )

    print("\nCreated workflow with:")
    print("  - Validation node (SPARC)")
    if LLM_CLIENT:
        print("  - Review node (Silent Review)")
    else:
        print("  - Review node (disabled - no LLM configured)")
    print("  - Proper routing based on validation/review results")

    # Run the workflow
    print("\nRunning workflow...")
    initial_state = {
        "messages": [{"role": "user", "content": "What's the weather in New York?"}],
    }
    result = app.invoke(initial_state)

    print("\nFinal state:")
    print(f"  Messages: {len(result.get('messages', []))} messages")
    print(f"  Validation: {result.get('validation_result')}")
    if LLM_CLIENT:
        print(f"  Review: {result.get('review_result')}")


# =============================================================================
# Example 4: Using Individual Nodes Standalone
# =============================================================================


def example_standalone_nodes():
    """Use ALTK nodes as standalone validators outside LangGraph."""
    print("\n" + "=" * 60)
    print("Example 4: Standalone Node Usage")
    print("=" * 60)

    # Create a validation node
    validator = create_validation_node(
        tool_specs=TOOL_SPECS,
        track="syntax",  # Fast syntax-only validation
    )

    # Validate a tool call directly
    state = {
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tool_specs": TOOL_SPECS,
        "current_tool_call": {
            "name": "get_weather",
            "arguments": {"city": "London", "units": "celsius"},
        },
    }

    print("\nValidating tool call...")
    result = validator(state)

    validation_result = result.get("validation_result")
    if validation_result:
        print(f"  Decision: {validation_result.decision}")
        print(f"  Should proceed: {validation_result.should_proceed}")
        if validation_result.issues:
            print(f"  Issues: {validation_result.issues}")

    # Now try with an invalid tool call
    print("\nValidating invalid tool call (missing required param)...")
    invalid_state = {
        "messages": [{"role": "user", "content": "Search for products"}],
        "tool_specs": TOOL_SPECS,
        "current_tool_call": {
            "name": "search_products",
            "arguments": {"max_results": 200},  # Missing 'query', invalid max_results
        },
    }

    result = validator(invalid_state)
    validation_result = result.get("validation_result")
    if validation_result:
        print(f"  Decision: {validation_result.decision}")
        if validation_result.issues:
            for issue in validation_result.issues:
                print(f"  Issue: {issue.explanation}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("ALTK LangGraph Integration Examples")
    print("====================================")

    # Show LLM configuration status
    if LLM_CLIENT:
        print("\nLLM client configured - review functionality enabled")
    else:
        print("\nNo LLM client configured - review functionality disabled")
        print("To enable, set: ALTK_LLM_PROVIDER and ALTK_MODEL_NAME")

    # Run examples

    try:
        example_manual_wiring()
    except Exception as e:
        print(f"Example 1 skipped: {e}")

    try:
        example_enhanced_graph()
    except Exception as e:
        print(f"Example 2 skipped: {e}")

    example_one_liner()

    try:
        example_standalone_nodes()
    except Exception as e:
        print(f"Example 4 skipped: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
