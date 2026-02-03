"""Prebuilt LangGraph agents with ALTK components.

This module provides ALTK-enhanced versions of LangGraph's prebuilt agents.
They extend the standard LangGraph agents with validation, review, and repair.

Example:
    from altk.integrations.langgraph.prebuilt import create_altk_react_agent

    # Same API as langgraph.prebuilt.create_react_agent, but with ALTK
    agent = create_altk_react_agent(
        model=ChatOpenAI(model="gpt-4"),
        tools=my_tools,
        validation_track="fast_track",
    )

    result = agent.invoke({"messages": [("user", "What's the weather?")]})
"""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    Union,
    Annotated,
)
import operator

logger = logging.getLogger(__name__)

# Type aliases
TrackType = Literal["syntax", "fast_track", "slow_track", "spec_free", "transformations_only"]


def _check_langgraph_installed():
    """Check if LangGraph is installed."""
    try:
        import langgraph
        return True
    except ImportError:
        raise ImportError(
            "langgraph is required for prebuilt agents. "
            "Install with: pip install langgraph langchain-core"
        )


# =============================================================================
# ALTK-Enhanced ReAct Agent (Extends LangGraph's create_react_agent)
# =============================================================================

def create_altk_react_agent(
    model: Any,
    tools: Sequence[Any],
    *,
    # ALTK-specific options
    validation_enabled: bool = True,
    review_enabled: bool = True,
    repair_enabled: bool = False,
    validation_track: TrackType = "fast_track",
    docs_path: Optional[str] = None,
    policies_path: Optional[str] = None,
    max_repair_attempts: int = 3,
    on_validation_fail: Literal["reject", "warn", "ignore"] = "reject",
    on_review_fail: Literal["repair", "warn", "ignore"] = "warn",
    altk_llm_client: Optional[Any] = None,
    # Standard LangGraph create_react_agent options (passed through)
    state_schema: Optional[type] = None,
    messages_modifier: Optional[Union[str, Callable]] = None,
    state_modifier: Optional[Callable] = None,
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
) -> Any:
    """Create an ALTK-enhanced ReAct agent.

    This extends LangGraph's `create_react_agent` with ALTK validation,
    review, and repair capabilities. All standard LangGraph options are
    supported and passed through.

    Args:
        model: The chat model to use (e.g., ChatOpenAI, ChatAnthropic)
        tools: List of tools the agent can use

        # ALTK-specific options
        validation_enabled: Enable pre-tool validation (default: True)
        review_enabled: Enable post-tool review (default: True)
        repair_enabled: Enable tool repair on failure (default: False)
        validation_track: SPARC validation track (default: "fast_track")
            - "syntax": Fast static validation only (no LLM)
            - "fast_track": 2 LLM calls
            - "slow_track": 5+ LLM calls (comprehensive)
            - "spec_free": Semantic without specs
        docs_path: Path to docs for RAG repair (required if repair_enabled)
        policies_path: Path to policy files for ToolGuard
        max_repair_attempts: Max repair attempts (default: 3)
        on_validation_fail: "reject" | "warn" | "ignore" (default: "reject")
        on_review_fail: "repair" | "warn" | "ignore" (default: "warn")
        altk_llm_client: Custom LLM client for ALTK components

        # Standard LangGraph options (passed to create_react_agent)
        state_schema: Custom state schema
        messages_modifier: System prompt or message modifier
        state_modifier: Function to modify state before model call
        checkpointer: LangGraph checkpointer for persistence
        interrupt_before: Nodes to interrupt before
        interrupt_after: Nodes to interrupt after
        debug: Enable debug mode
        version: LangGraph version ("v1" or "v2")

    Returns:
        Compiled LangGraph agent with ALTK enhancements

    Example:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from altk.integrations.langgraph import create_altk_react_agent

        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: 72°F, sunny"

        # Create ALTK-enhanced agent
        agent = create_altk_react_agent(
            model=ChatOpenAI(model="gpt-4"),
            tools=[get_weather],
            validation_track="fast_track",
            on_validation_fail="reject",
        )

        result = agent.invoke({
            "messages": [("user", "What's the weather in NYC?")]
        })
    """
    _check_langgraph_installed()

    # If no ALTK features enabled, just return standard react agent
    if not validation_enabled and not review_enabled and not repair_enabled:
        try:
            from langgraph.prebuilt import create_react_agent  # type: ignore[import-untyped]
            return create_react_agent(
                model=model,
                tools=tools,
                state_schema=state_schema,
                messages_modifier=messages_modifier,
                state_modifier=state_modifier,
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
        except ImportError:
            # Fall through to build our own
            pass

    # Build ALTK-enhanced agent using custom graph
    # We need to build our own graph to inject ALTK nodes
    return _build_altk_react_agent(
        model=model,
        tools=tools,
        validation_enabled=validation_enabled,
        review_enabled=review_enabled,
        repair_enabled=repair_enabled,
        validation_track=validation_track,
        docs_path=docs_path,
        policies_path=policies_path,
        max_repair_attempts=max_repair_attempts,
        on_validation_fail=on_validation_fail,
        on_review_fail=on_review_fail,
        altk_llm_client=altk_llm_client,
        state_schema=state_schema,
        messages_modifier=messages_modifier,
        state_modifier=state_modifier,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


def _build_altk_react_agent(
    model: Any,
    tools: Sequence[Any],
    validation_enabled: bool,
    review_enabled: bool,
    repair_enabled: bool,
    validation_track: TrackType,
    docs_path: Optional[str],
    policies_path: Optional[str],
    max_repair_attempts: int,
    on_validation_fail: str,
    on_review_fail: str,
    altk_llm_client: Optional[Any],
    state_schema: Optional[type],
    messages_modifier: Optional[Union[str, Callable]],
    state_modifier: Optional[Callable],
    checkpointer: Optional[Any],
    interrupt_before: Optional[List[str]],
    interrupt_after: Optional[List[str]],
    debug: bool,
) -> Any:
    """Build the ALTK-enhanced ReAct agent graph."""
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

    from altk.integrations.langgraph.nodes import (
        create_validation_node,
        create_review_node,
        create_repair_node,
    )
    from altk.integrations.langgraph.types import (
        ValidationDecision,
        ReviewOutcome,
    )

    # Convert tools to specs for validation
    tool_specs = _tools_to_specs(tools)

    # Define state schema
    class ALTKReActState(TypedDict, total=False):
        messages: Annotated[List[Any], operator.add]
        # ALTK fields
        tool_specs: List[Dict[str, Any]]
        current_tool_call: Optional[Dict[str, Any]]
        tool_response: Optional[Any]
        validation_result: Optional[Any]
        review_result: Optional[Any]
        repair_result: Optional[Any]
        repair_attempts: int

    State = state_schema or ALTKReActState

    # Create model with tools
    model_with_tools = model.bind_tools(tools)

    # Create ALTK nodes
    validation_node = None
    review_node = None
    repair_node = None

    if validation_enabled:
        validation_node = create_validation_node(
            tool_specs=tool_specs,
            track=validation_track,
            llm_client=altk_llm_client,
            policies_path=policies_path,
        )

    if review_enabled:
        review_node = create_review_node(
            llm_client=altk_llm_client,
            review_type="json",
        )

    if repair_enabled and docs_path:
        repair_node = create_repair_node(
            docs_path=docs_path,
            llm_client=altk_llm_client,
        )

    # Define agent node (similar to LangGraph's internal implementation)
    def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])

        # Apply messages_modifier (system prompt)
        if messages_modifier:
            if callable(messages_modifier):
                messages = messages_modifier(messages)
            elif isinstance(messages_modifier, str):
                messages = [SystemMessage(content=messages_modifier)] + list(messages)

        # Apply state_modifier if provided
        if state_modifier:
            messages = state_modifier(state)

        # Check for validation rejection - add feedback
        validation_result = state.get("validation_result")
        if validation_result and validation_result.decision == ValidationDecision.REJECT:
            if on_validation_fail == "reject":
                issues = [i.explanation for i in validation_result.issues]
                feedback = f"Your previous tool call was rejected due to validation errors: {issues}. Please fix and try again."
                messages = list(messages) + [SystemMessage(content=feedback)]

        # Call the model
        response = model_with_tools.invoke(messages)

        # Extract tool call info for ALTK
        current_tool_call = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tc = response.tool_calls[0]
            current_tool_call = {
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
                "id": tc.get("id", ""),
            }

        return {
            "messages": [response],
            "current_tool_call": current_tool_call,
            "tool_specs": tool_specs,
            "validation_result": None,  # Reset for new tool call
            "review_result": None,
        }

    # Define tools node
    def call_tools(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {}

        # Check if we should use corrected tool call from validation
        validation_result = state.get("validation_result")
        tool_calls = list(last_message.tool_calls)

        if validation_result and validation_result.has_corrections:
            corrected = validation_result.corrected_tool_call
            tool_calls = [{
                **tool_calls[0],
                "args": corrected.arguments,
            }]
            logger.info(f"Using corrected tool call: {corrected.arguments}")

        # Execute tools using LangGraph's ToolNode
        tool_executor = ToolNode(tools)

        # Create message with potentially corrected tool calls
        exec_message = AIMessage(content="", tool_calls=tool_calls)
        result = tool_executor.invoke({"messages": [exec_message]})

        # Extract response for review
        tool_response = None
        result_messages = result.get("messages", [])
        if result_messages:
            last_tool_msg = result_messages[-1]
            if isinstance(last_tool_msg, ToolMessage):
                tool_response = last_tool_msg.content

        return {
            "messages": result_messages,
            "tool_response": tool_response,
        }

    # Define routing functions
    def should_continue(state: Dict[str, Any]) -> str:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message:
            return "end"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "validate" if validation_enabled else "tools"

        return "end"

    def route_after_validation(state: Dict[str, Any]) -> str:
        validation_result = state.get("validation_result")

        if validation_result is None:
            return "tools"

        decision = validation_result.decision

        if decision == ValidationDecision.APPROVE:
            return "tools"

        # Handle rejection based on config
        if on_validation_fail == "ignore":
            return "tools"
        elif on_validation_fail == "warn":
            logger.warning(f"Validation issues (proceeding anyway): {validation_result.issues}")
            return "tools"
        else:  # reject
            logger.info(f"Validation rejected tool call: {validation_result.issues}")
            return "agent"

    def route_after_review(state: Dict[str, Any]) -> str:
        review_result = state.get("review_result")

        if review_result is None:
            return "agent"

        if review_result.outcome == ReviewOutcome.ACCOMPLISHED:
            return "agent"

        # Handle failure based on config
        if review_result.needs_repair and repair_enabled and repair_node:
            repair_attempts = state.get("repair_attempts", 0)
            if repair_attempts < max_repair_attempts:
                return "repair"

        if on_review_fail == "warn":
            logger.warning(f"Review detected issues: {review_result.details}")

        return "agent"

    def route_after_repair(state: Dict[str, Any]) -> str:
        repair_result = state.get("repair_result")

        if repair_result and repair_result.success:
            return "tools"

        return "agent"

    def increment_repair_attempts(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"repair_attempts": state.get("repair_attempts", 0) + 1}

    # Build the graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    if validation_node:
        workflow.add_node("validate", validation_node)

    if review_node:
        workflow.add_node("review", review_node)

    if repair_node:
        def repair_with_counter(state):
            result = repair_node(state)
            result["repair_attempts"] = state.get("repair_attempts", 0) + 1
            return result
        workflow.add_node("repair", repair_with_counter)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges: agent -> validate/tools/end
    if validation_enabled:
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
    else:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )

    # Add edges: tools -> review/agent
    if review_enabled:
        workflow.add_edge("tools", "review")

        if repair_enabled and repair_node:
            workflow.add_conditional_edges(
                "review",
                route_after_review,
                {"agent": "agent", "repair": "repair"}
            )
            workflow.add_conditional_edges(
                "repair",
                route_after_repair,
                {"tools": "tools", "agent": "agent"}
            )
        else:
            workflow.add_edge("review", "agent")
    else:
        workflow.add_edge("tools", "agent")

    # Compile with standard LangGraph options
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


# =============================================================================
# Convenience Wrappers
# =============================================================================

def create_altk_safe_agent(
    model: Any,
    tools: Sequence[Any],
    *,
    policies_path: Optional[str] = None,
    docs_path: Optional[str] = None,
    messages_modifier: Optional[Union[str, Callable]] = None,
    checkpointer: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Create a maximally safe agent with all ALTK protections.

    Features:
    - Full semantic validation (slow_track)
    - ToolGuard policy enforcement (if policies_path provided)
    - Silent error detection via review
    - Automatic RAG repair (if docs_path provided)

    Use for high-stakes applications where reliability is critical.

    Args:
        model: The chat model to use
        tools: List of tools
        policies_path: Path to ToolGuard policy files
        docs_path: Path to documentation for RAG repair
        messages_modifier: System prompt or message modifier
        checkpointer: LangGraph checkpointer
        **kwargs: Additional arguments passed to create_altk_react_agent

    Returns:
        Compiled LangGraph agent
    """
    return create_altk_react_agent(
        model=model,
        tools=tools,
        validation_enabled=True,
        review_enabled=True,
        repair_enabled=docs_path is not None,
        validation_track="slow_track",
        policies_path=policies_path,
        docs_path=docs_path,
        max_repair_attempts=3,
        on_validation_fail="reject",
        on_review_fail="repair" if docs_path else "warn",
        messages_modifier=messages_modifier,
        checkpointer=checkpointer,
        **kwargs,
    )


def create_altk_fast_agent(
    model: Any,
    tools: Sequence[Any],
    *,
    messages_modifier: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> Any:
    """Create a fast agent with minimal validation overhead.

    Features:
    - Syntax-only validation (no LLM calls for validation)
    - No review step
    - Minimal latency overhead

    Use when speed is more important than comprehensive validation.

    Args:
        model: The chat model to use
        tools: List of tools
        messages_modifier: System prompt or message modifier
        **kwargs: Additional arguments passed to create_altk_react_agent

    Returns:
        Compiled LangGraph agent
    """
    return create_altk_react_agent(
        model=model,
        tools=tools,
        validation_enabled=True,
        review_enabled=False,
        repair_enabled=False,
        validation_track="syntax",
        on_validation_fail="warn",
        messages_modifier=messages_modifier,
        **kwargs,
    )


def create_altk_tool_agent(
    model: Any,
    tools: Sequence[Any],
    *,
    validation_track: TrackType = "syntax",
    review_enabled: bool = True,
    messages_modifier: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> Any:
    """Create a simple tool-calling agent with ALTK validation.

    A balanced agent with basic validation and review but no repair.

    Args:
        model: The chat model to use
        tools: List of tools
        validation_track: Validation track (default: "syntax")
        review_enabled: Enable post-tool review (default: True)
        messages_modifier: System prompt or message modifier
        **kwargs: Additional arguments passed to create_altk_react_agent

    Returns:
        Compiled LangGraph agent
    """
    return create_altk_react_agent(
        model=model,
        tools=tools,
        validation_enabled=True,
        review_enabled=review_enabled,
        repair_enabled=False,
        validation_track=validation_track,
        on_validation_fail="warn",
        on_review_fail="warn",
        messages_modifier=messages_modifier,
        **kwargs,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _tools_to_specs(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain tools to OpenAI-style tool specifications."""
    specs = []

    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            # LangChain BaseTool
            spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _get_tool_schema(tool),
                },
            }
            specs.append(spec)
        elif isinstance(tool, dict):
            # Already a spec dict
            specs.append(tool)
        elif callable(tool) and hasattr(tool, "__name__"):
            # Plain function - try to extract info
            spec = {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            specs.append(spec)

    return specs


def _get_tool_schema(tool: Any) -> Dict[str, Any]:
    """Extract JSON schema from a LangChain tool."""
    # Try args_schema (Pydantic model)
    if hasattr(tool, "args_schema") and tool.args_schema:
        try:
            return tool.args_schema.model_json_schema()
        except Exception:
            pass

    # Try schema property
    if hasattr(tool, "schema"):
        try:
            schema = tool.schema
            if callable(schema):
                return schema()
            return schema
        except Exception:
            pass

    # Try get_input_schema
    if hasattr(tool, "get_input_schema"):
        try:
            return tool.get_input_schema().model_json_schema()
        except Exception:
            pass

    # Fallback
    return {"type": "object", "properties": {}}


# =============================================================================
# One-liner Workflow Builder
# =============================================================================

def create_altk_workflow(
    agent_node: Callable[[Dict[str, Any]], Dict[str, Any]],
    tool_node: Callable[[Dict[str, Any]], Dict[str, Any]],
    tool_specs: List[Dict[str, Any]],
    *,
    validation_enabled: bool = True,
    review_enabled: bool = True,
    repair_enabled: bool = False,
    validation_track: TrackType = "fast_track",
    docs_path: Optional[str] = None,
    policies_path: Optional[str] = None,
    max_repair_attempts: int = 3,
    on_validation_fail: Literal["reject", "warn", "ignore"] = "reject",
    on_review_fail: Literal["repair", "warn", "ignore"] = "warn",
    altk_llm_client: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    debug: bool = False,
) -> Any:
    """Create a fully ALTK-enhanced workflow from raw node functions.

    This is the simplest way to create an ALTK workflow when you already
    have agent and tool node functions defined.

    Args:
        agent_node: Function that takes state and returns agent decisions
            (must populate 'current_tool_call' when calling tools)
        tool_node: Function that takes state and executes the tool
            (must populate 'tool_response' with the result)
        tool_specs: List of OpenAI-style tool specifications

        # ALTK options
        validation_enabled: Enable pre-tool validation (default: True)
        review_enabled: Enable post-tool review (default: True)
        repair_enabled: Enable tool repair on failure (default: False)
        validation_track: SPARC validation track (default: "fast_track")
        docs_path: Path to docs for RAG repair (required if repair_enabled)
        policies_path: Path to policy files for ToolGuard
        max_repair_attempts: Max repair attempts (default: 3)
        on_validation_fail: "reject" | "warn" | "ignore" (default: "reject")
        on_review_fail: "repair" | "warn" | "ignore" (default: "warn")
        altk_llm_client: Custom LLM client for ALTK components

        # LangGraph options
        checkpointer: LangGraph checkpointer for persistence
        debug: Enable debug mode

    Returns:
        Compiled LangGraph workflow with ALTK enhancements

    Example:
        def agent_node(state):
            return {
                "messages": [...],
                "current_tool_call": {
                    "name": "get_weather",
                    "arguments": {"city": "NYC"},
                },
            }

        def tool_node(state):
            return {
                "tool_response": {"temperature": 72},
            }

        app = create_altk_workflow(
            agent_node=agent_node,
            tool_node=tool_node,
            tool_specs=my_tool_specs,
            validation_track="fast_track",
            review_enabled=True,
        )

        result = app.invoke({"messages": [...]})
    """
    _check_langgraph_installed()

    from langgraph.graph import StateGraph, END
    from altk.integrations.langgraph.nodes import (
        create_validation_node,
        create_review_node,
        create_repair_node,
    )
    from altk.integrations.langgraph.types import (
        ValidationDecision,
        ReviewOutcome,
    )

    # Define state schema
    class ALTKWorkflowState(TypedDict, total=False):
        messages: Annotated[List[Any], operator.add]
        tool_specs: List[Dict[str, Any]]
        current_tool_call: Optional[Dict[str, Any]]
        tool_response: Optional[Any]
        validation_result: Optional[Any]
        review_result: Optional[Any]
        repair_result: Optional[Any]
        repair_attempts: int

    # Create ALTK nodes
    validation_node = None
    review_node = None
    repair_node = None

    if validation_enabled:
        validation_node = create_validation_node(
            tool_specs=tool_specs,
            track=validation_track,
            llm_client=altk_llm_client,
            policies_path=policies_path,
        )

    if review_enabled and altk_llm_client:
        review_node = create_review_node(
            llm_client=altk_llm_client,
            review_type="json",
        )

    if repair_enabled and docs_path and altk_llm_client:
        repair_node = create_repair_node(
            docs_path=docs_path,
            llm_client=altk_llm_client,
        )

    # Wrap agent node to inject tool_specs
    def wrapped_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        result = agent_node(state)
        result["tool_specs"] = tool_specs
        # Reset ALTK state for new tool call
        result["validation_result"] = None
        result["review_result"] = None
        return result

    # Define routing functions
    def should_continue(state: Dict[str, Any]) -> str:
        current_tool_call = state.get("current_tool_call")
        if current_tool_call:
            return "validate" if validation_enabled else "tools"
        return "end"

    def route_after_validation(state: Dict[str, Any]) -> str:
        validation_result = state.get("validation_result")

        if validation_result is None:
            return "tools"

        decision = validation_result.decision

        if decision == ValidationDecision.APPROVE:
            return "tools"

        if on_validation_fail == "ignore":
            return "tools"
        elif on_validation_fail == "warn":
            logger.warning(f"Validation issues (proceeding anyway): {validation_result.issues}")
            return "tools"
        else:  # reject
            logger.info(f"Validation rejected tool call: {validation_result.issues}")
            return "agent"

    def route_after_review(state: Dict[str, Any]) -> str:
        review_result = state.get("review_result")

        if review_result is None:
            return "agent"

        if review_result.outcome == ReviewOutcome.ACCOMPLISHED:
            return "agent"

        if review_result.needs_repair and repair_enabled and repair_node:
            repair_attempts = state.get("repair_attempts", 0)
            if repair_attempts < max_repair_attempts:
                return "repair"

        if on_review_fail == "warn":
            logger.warning(f"Review detected issues: {review_result.details}")

        return "agent"

    def route_after_repair(state: Dict[str, Any]) -> str:
        repair_result = state.get("repair_result")
        if repair_result and repair_result.success:
            return "tools"
        return "agent"

    # Build the graph
    workflow = StateGraph(ALTKWorkflowState)

    # Add nodes
    workflow.add_node("agent", wrapped_agent_node)
    workflow.add_node("tools", tool_node)

    if validation_node:
        workflow.add_node("validate", validation_node)

    if review_node:
        workflow.add_node("review", review_node)

    if repair_node:
        def repair_with_counter(state):
            result = repair_node(state)
            result["repair_attempts"] = state.get("repair_attempts", 0) + 1
            return result
        workflow.add_node("repair", repair_with_counter)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    if validation_enabled:
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
    else:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )

    # Tools -> Review/End or Agent
    if review_node:
        workflow.add_edge("tools", "review")

        if repair_enabled and repair_node:
            workflow.add_conditional_edges(
                "review",
                route_after_review,
                {"agent": "agent", "repair": "repair"}
            )
            workflow.add_conditional_edges(
                "repair",
                route_after_repair,
                {"tools": "tools", "agent": "agent"}
            )
        else:
            # End after review (no repair loop)
            workflow.add_edge("review", END)
    else:
        # No review, end after tools
        workflow.add_edge("tools", END)

    # Compile
    return workflow.compile(
        checkpointer=checkpointer,
        debug=debug,
    )
