"""
Simple LangGraph Agent with Tools, RAG, and ALTK Enhancements

This example demonstrates how to create a LangGraph-based agent enhanced with
ALTK components for improved reliability:

- SPARC: Pre-tool validation to ensure tool calls are correct
- Silent Review: Post-tool review to detect silent errors in responses

The agent can answer questions using tools and a knowledge base (RAG).
"""

import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ALTK imports (for reference - we implement simplified versions inline)
# from altk.pre_tool.sparc import SPARCReflectionComponent
# from altk.pre_tool.core import Track
# from altk.post_tool.silent_review.silent_review import SilentReviewForJSONDataComponent
# from altk.post_tool.core.toolkit import SilentReviewRunInput, Outcome
# from altk.core.toolkit import AgentPhase, ComponentConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ollama runs locally - no API key needed
# Make sure Ollama is running: ollama serve
# And pull a model: ollama pull llama3.2

# Simple in-memory document store for RAG
KNOWLEDGE_BASE = {
    "company_policy": """
    Company Policy Document:
    - Work hours: 9 AM to 5 PM, Monday through Friday
    - Remote work: Allowed 2 days per week with manager approval
    - Vacation: 15 days per year for new employees, 20 days after 3 years
    - Sick leave: 10 days per year
    - Equipment: Laptop and monitor provided, $500 home office stipend
    """,
    "product_info": """
    Product Information:
    - Product Name: SuperWidget Pro
    - Price: $99.99 (Basic), $199.99 (Pro), $499.99 (Enterprise)
    - Features: Real-time sync, cloud backup, team collaboration
    - Support: Email support for Basic, 24/7 phone for Pro and Enterprise
    - Trial: 14-day free trial available for all plans
    """,
    "technical_docs": """
    Technical Documentation:
    - API Rate Limits: 100 requests/minute (Basic), 1000/minute (Pro)
    - Supported formats: JSON, XML, CSV
    - Authentication: OAuth 2.0 and API keys supported
    - Webhooks: Available for Pro and Enterprise plans
    - SDK: Python, JavaScript, Java, and Go SDKs available
    """,
}


# ============================================================================
# TOOLS
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    Use this tool when you need to find information about company policies,
    products, or technical documentation.

    Args:
        query: The search query describing what information you need
    """
    query_lower = query.lower()
    results = []

    if any(word in query_lower for word in ["policy", "work", "vacation", "remote", "sick", "equipment"]):
        results.append(KNOWLEDGE_BASE["company_policy"])

    if any(word in query_lower for word in ["product", "price", "feature", "support", "trial", "plan"]):
        results.append(KNOWLEDGE_BASE["product_info"])

    if any(word in query_lower for word in ["api", "technical", "rate", "format", "auth", "sdk", "webhook"]):
        results.append(KNOWLEDGE_BASE["technical_docs"])

    if not results:
        results = list(KNOWLEDGE_BASE.values())

    return "\n\n---\n\n".join(results)


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    Use this for any math operations like addition, multiplication, percentages, etc.

    Args:
        expression: A mathematical expression to evaluate (e.g., "100 * 0.15" or "50 + 30")
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators allowed."
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def get_current_date() -> str:
    """
    Get the current date and time.
    Use this when you need to know today's date or current time.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============================================================================
# TOOL SPECIFICATIONS (for ALTK validation)
# ============================================================================

def get_tool_specs(tools):
    """Convert LangChain tools to OpenAI-style tool specifications."""
    specs = []
    for t in tools:
        spec = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}},
            },
        }
        specs.append(spec)
    return specs


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict, total=False):
    """State for the ALTK-enhanced agent."""
    messages: Annotated[List[Any], operator.add]
    current_tool_call: Optional[Dict[str, Any]]
    tool_response: Optional[str]
    validation_passed: bool
    validation_issues: List[str]
    review_outcome: Optional[str]
    review_details: Optional[str]


# ============================================================================
# ALTK-ENHANCED AGENT
# ============================================================================

def create_rag_agent_altk():
    """
    Create a LangGraph agent with explicit ALTK component integration.

    This builds a custom graph with:
    - Agent node: LLM decides what to do
    - SPARC Validation node: Validates tool calls before execution
    - Tool execution node: Runs the tools
    - Silent Review node: Reviews tool outputs for errors

    Returns:
        Compiled LangGraph agent
    """

    # Initialize the LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    # Define tools
    tools = [search_knowledge_base, calculate, get_current_date]
    tool_specs = get_tool_specs(tools)

    # Bind tools to the model
    model_with_tools = llm.bind_tools(tools)

    # Create tool executor
    tool_executor = ToolNode(tools)

    # System prompt
    system_prompt = """You are a helpful assistant that can answer questions using available tools.

When answering questions:
1. If the question is about company policies, products, or technical information,
   use the search_knowledge_base tool to find relevant information.
2. If the question involves calculations, use the calculate tool.
3. If the question involves dates or times, use the get_current_date tool.

Always base your answers on the information retrieved from tools when available.
Be concise and helpful in your responses."""

    # ========================================================================
    # NODE: Agent (LLM decides what to do)
    # ========================================================================
    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])

        # Add system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        # Call the model
        response = model_with_tools.invoke(messages)

        # Extract tool call info if present
        current_tool_call = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tc = response.tool_calls[0]
            current_tool_call = {
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
                "id": tc.get("id", ""),
            }
            print(f"    [Agent] Calling tool: {current_tool_call['name']}({current_tool_call['arguments']})")
        else:
            print(f"    [Agent] Final response ready")

        return {
            "messages": [response],
            "current_tool_call": current_tool_call,
            "validation_passed": True,
            "validation_issues": [],
        }

    # ========================================================================
    # NODE: SPARC Validation (Pre-tool validation)
    # ========================================================================
    def sparc_validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        SPARC validation node - validates tool calls before execution.

        SPARC checks:
        - Tool name matches available tools
        - Arguments match the expected schema
        - Required parameters are present
        - Argument types are correct
        """
        print("    [SPARC] Validating tool call...")

        tool_call = state.get("current_tool_call")
        if not tool_call:
            return {"validation_passed": True, "validation_issues": []}

        issues = []

        # Check if tool exists
        tool_names = [t.name for t in tools]
        if tool_call["name"] not in tool_names:
            issues.append(f"Unknown tool: {tool_call['name']}")

        # Find the tool spec and validate arguments
        for spec in tool_specs:
            if spec["function"]["name"] == tool_call["name"]:
                params = spec["function"].get("parameters", {})
                required = params.get("required", [])
                properties = params.get("properties", {})

                # Check required parameters
                for req in required:
                    if req not in tool_call["arguments"]:
                        issues.append(f"Missing required parameter: {req}")

                # Check for unknown parameters
                for arg in tool_call["arguments"]:
                    if arg not in properties:
                        issues.append(f"Unknown parameter: {arg}")

                break

        if issues:
            print(f"    [SPARC] Validation FAILED: {issues}")
            return {"validation_passed": False, "validation_issues": issues}
        else:
            print(f"    [SPARC] Validation PASSED")
            return {"validation_passed": True, "validation_issues": []}

    # ========================================================================
    # NODE: Tool Execution
    # ========================================================================
    def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool and capture the response."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {}

        print(f"    [Tool] Executing tool...")

        # Execute the tool
        result = tool_executor.invoke({"messages": [last_message]})

        # Extract tool response
        tool_response = None
        result_messages = result.get("messages", [])
        if result_messages:
            tool_response = result_messages[-1].content
            print(f"    [Tool] Got response: {tool_response[:100]}...")

        return {
            "messages": result_messages,
            "tool_response": tool_response,
        }

    # ========================================================================
    # NODE: Silent Review (Post-tool review)
    # ========================================================================
    def silent_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Silent Review node - detects silent errors in tool responses.

        Silent errors are when a tool returns a response that looks successful
        but actually contains errors, missing data, or irrelevant information.
        """
        print("    [Silent Review] Reviewing tool response...")

        tool_response = state.get("tool_response", "")
        tool_call = state.get("current_tool_call", {})

        # Simple heuristic-based review (in production, use LLM-based review)
        issues = []

        # Check for error indicators in response
        error_indicators = ["error", "failed", "not found", "invalid", "exception"]
        response_lower = tool_response.lower() if tool_response else ""

        for indicator in error_indicators:
            if indicator in response_lower:
                issues.append(f"Response may contain error: '{indicator}' found")

        # Check for empty or very short responses
        if not tool_response or len(tool_response.strip()) < 10:
            issues.append("Response is empty or too short")

        if issues:
            print(f"    [Silent Review] Issues detected: {issues}")
            return {
                "review_outcome": "issues_detected",
                "review_details": "; ".join(issues),
            }
        else:
            print(f"    [Silent Review] Response looks good")
            return {
                "review_outcome": "ok",
                "review_details": None,
            }

    # ========================================================================
    # ROUTING FUNCTIONS
    # ========================================================================
    def should_continue(state: Dict[str, Any]) -> str:
        """Decide whether to validate, or end."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message:
            return "end"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "validate"

        return "end"

    def route_after_validation(state: Dict[str, Any]) -> str:
        """Route based on validation result."""
        if state.get("validation_passed", True):
            return "tools"
        else:
            # On validation failure, go back to agent with feedback
            return "agent"

    # ========================================================================
    # BUILD THE GRAPH
    # ========================================================================
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", sparc_validation_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("review", silent_review_node)

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
    workflow.add_edge("review", "agent")

    # Compile
    return workflow.compile()


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def ask_question(agent, question: str) -> str:
    """Ask the agent a question and get a response."""
    print(f"\n  Processing...")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    final_message = result["messages"][-1]
    return final_message.content


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Simple RAG Agent with ALTK Components - Demo")
    print("=" * 60)
    print("\nALTK Components in use:")
    print("  - SPARC Validation: Pre-tool validation")
    print("  - Silent Review: Post-tool error detection")
    print("\nNote: Make sure Ollama is running (ollama serve)")
    print("And you have a model pulled (ollama pull llama3.2)")

    # Create the agent
    agent = create_rag_agent_altk()

    # Test questions
    questions = [
        "What is the vacation policy for new employees?",
        "How much does the Pro plan cost?",
        "What is 15% of $199.99?",
        "What are the API rate limits for the Pro plan?",
        "What date is it today?",
        "Can I work remotely? What's the policy?",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}")
        response = ask_question(agent, q)
        print(f"\nA: {response}")
