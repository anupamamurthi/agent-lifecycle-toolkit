"""LangGraph integration for ALTK.

This module provides LangGraph-compatible nodes that wrap ALTK components,
allowing you to easily add validation, review, and repair capabilities to
your LangGraph agents.

Example:
    from langgraph.graph import StateGraph
    from altk.integrations.langgraph import (
        ALTKState,
        create_validation_node,
        create_review_node,
        create_repair_node,
        ValidationDecision,
        ReviewOutcome,
    )

    workflow = StateGraph(ALTKState)
    workflow.add_node("validate", create_validation_node(tool_specs=my_tools))
    workflow.add_node("review", create_review_node())
    workflow.add_node("repair", create_repair_node(docs_path="./docs"))
"""

from altk.integrations.langgraph.types import (
    ALTKState,
    ToolCallInfo,
    ValidationResult,
    ReviewResult,
    RepairResult,
    ValidationDecision,
    ReviewOutcome,
)
from altk.integrations.langgraph.nodes import (
    create_validation_node,
    create_review_node,
    create_repair_node,
    ValidationNode,
    ReviewNode,
    RepairNode,
)
from altk.integrations.langgraph.enhanced_graph import (
    ALTKEnhancedGraph,
    ALTKGraphConfig,
)
from altk.integrations.langgraph.prebuilt import (
    create_altk_react_agent,
    create_altk_tool_agent,
    create_altk_safe_agent,
    create_altk_fast_agent,
    create_altk_workflow,
)

__all__ = [
    # Types
    "ALTKState",
    "ToolCallInfo",
    "ValidationResult",
    "ReviewResult",
    "RepairResult",
    "ValidationDecision",
    "ReviewOutcome",
    # Node factories
    "create_validation_node",
    "create_review_node",
    "create_repair_node",
    # Node classes
    "ValidationNode",
    "ReviewNode",
    "RepairNode",
    # Enhanced graph
    "ALTKEnhancedGraph",
    "ALTKGraphConfig",
    # Prebuilt agents
    "create_altk_react_agent",
    "create_altk_tool_agent",
    "create_altk_safe_agent",
    "create_altk_fast_agent",
    # Workflow builder
    "create_altk_workflow",
]
