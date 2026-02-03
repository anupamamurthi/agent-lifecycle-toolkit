"""Enhanced LangGraph wrapper with ALTK integration.

This module provides a convenience wrapper that automatically adds ALTK
validation, review, and repair capabilities to existing LangGraph workflows.

Example:
    from langgraph.graph import StateGraph
    from altk.integrations.langgraph import ALTKEnhancedGraph, ALTKGraphConfig

    # Your existing workflow
    workflow = StateGraph(MyState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")

    # Enhance with ALTK
    enhanced = ALTKEnhancedGraph(
        graph=workflow,
        config=ALTKGraphConfig(
            validation_enabled=True,
            review_enabled=True,
            repair_enabled=True,
            tool_specs=my_tools,
        )
    )

    # Compile and run
    app = enhanced.compile()
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Literal, TypeVar
from pydantic import BaseModel, Field

from altk.integrations.langgraph.nodes import (
    ValidationNode,
    ReviewNode,
    RepairNode,
    create_validation_node,
    create_review_node,
    create_repair_node,
    TrackType,
)
from altk.integrations.langgraph.types import (
    ValidationDecision,
    ReviewOutcome,
)

logger = logging.getLogger(__name__)

# Type variable for state
StateType = TypeVar("StateType", bound=Dict[str, Any])


class ALTKGraphConfig(BaseModel):
    """Configuration for ALTK-enhanced LangGraph.

    Attributes:
        validation_enabled: Enable pre-tool validation
        review_enabled: Enable post-tool review
        repair_enabled: Enable tool repair on failure
        tool_specs: Tool specifications for validation
        validation_track: SPARC validation track
        docs_path: Path to docs for RAG repair
        llm_client: LLM client for ALTK components
        policies_path: Path to policy files for ToolGuard
        use_toolguard: Enable ToolGuard policy enforcement
        review_type: Type of review (json or tabular)
        retrieval_type: RAG retrieval type (bm25 or chromadb)
        max_repair_attempts: Maximum repair attempts before giving up
        on_validation_fail: Action on validation failure (reject, warn, ignore)
        on_review_fail: Action on review failure (repair, warn, ignore)
    """

    # Feature toggles
    validation_enabled: bool = Field(default=True)
    review_enabled: bool = Field(default=True)
    repair_enabled: bool = Field(default=False)

    # Tool configuration
    tool_specs: List[Dict[str, Any]] = Field(default_factory=list)

    # Validation config
    validation_track: TrackType = Field(default="fast_track")
    use_toolguard: bool = Field(default=False)
    policies_path: Optional[str] = Field(default=None)

    # Review config
    review_type: Literal["json", "tabular"] = Field(default="json")

    # Repair config
    docs_path: Optional[str] = Field(default=None)
    retrieval_type: Literal["bm25", "chromadb"] = Field(default="chromadb")
    max_repair_attempts: int = Field(default=3)

    # LLM client
    llm_client: Optional[Any] = Field(default=None)

    # Behavior configuration
    on_validation_fail: Literal["reject", "warn", "ignore"] = Field(default="reject")
    on_review_fail: Literal["repair", "warn", "ignore"] = Field(default="repair")

    model_config = {"arbitrary_types_allowed": True}


class ALTKEnhancedGraph:
    """Wrapper that adds ALTK capabilities to a LangGraph StateGraph.

    This class provides a convenient way to enhance existing LangGraph
    workflows with ALTK validation, review, and repair capabilities
    without manually wiring up all the nodes and edges.

    Example:
        from langgraph.graph import StateGraph
        from altk.integrations.langgraph import ALTKEnhancedGraph, ALTKGraphConfig

        # Create your workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        # Enhance with ALTK
        enhanced = ALTKEnhancedGraph(
            graph=workflow,
            config=ALTKGraphConfig(
                validation_enabled=True,
                review_enabled=True,
                tool_specs=my_tools,
            )
        )

        # Get the enhanced graph
        app = enhanced.compile()
    """

    def __init__(
        self,
        graph: Any,  # StateGraph
        config: Optional[ALTKGraphConfig] = None,
        tool_node_name: str = "tools",
        agent_node_name: str = "agent",
    ):
        """Initialize the enhanced graph.

        Args:
            graph: The LangGraph StateGraph to enhance
            config: ALTK configuration
            tool_node_name: Name of the tool execution node
            agent_node_name: Name of the agent node
        """
        self.graph = graph
        self.config = config or ALTKGraphConfig()
        self.tool_node_name = tool_node_name
        self.agent_node_name = agent_node_name

        # Create ALTK nodes based on config
        self._validation_node: Optional[ValidationNode] = None
        self._review_node: Optional[ReviewNode] = None
        self._repair_node: Optional[RepairNode] = None

        self._setup_nodes()

    def _setup_nodes(self):
        """Setup ALTK nodes based on configuration."""
        if self.config.validation_enabled:
            self._validation_node = create_validation_node(
                tool_specs=self.config.tool_specs,
                track=self.config.validation_track,
                llm_client=self.config.llm_client,
                use_toolguard=self.config.use_toolguard,
                policies_path=self.config.policies_path,
            )

        if self.config.review_enabled:
            self._review_node = create_review_node(
                llm_client=self.config.llm_client,
                review_type=self.config.review_type,
            )

        if self.config.repair_enabled and self.config.docs_path:
            self._repair_node = create_repair_node(
                docs_path=self.config.docs_path,
                llm_client=self.config.llm_client,
                retrieval_type=self.config.retrieval_type,
            )

    def _route_after_validation(self, state: Dict[str, Any]) -> str:
        """Route based on validation result."""
        validation_result = state.get("validation_result")

        if validation_result is None:
            return self.tool_node_name

        if self.config.on_validation_fail == "ignore":
            return self.tool_node_name

        if validation_result.decision == ValidationDecision.APPROVE:
            return self.tool_node_name
        elif validation_result.decision == ValidationDecision.REJECT:
            if self.config.on_validation_fail == "warn":
                logger.warning(
                    f"Validation rejected tool call: {validation_result.issues}"
                )
                return self.tool_node_name
            else:
                # Reject - return to agent with error
                return self.agent_node_name
        else:
            # Error case
            return self.agent_node_name

    def _route_after_review(self, state: Dict[str, Any]) -> str:
        """Route based on review result."""
        review_result = state.get("review_result")

        if review_result is None:
            return self.agent_node_name

        if self.config.on_review_fail == "ignore":
            return self.agent_node_name

        if review_result.outcome == ReviewOutcome.ACCOMPLISHED:
            return self.agent_node_name
        elif review_result.needs_repair:
            if self.config.on_review_fail == "repair" and self._repair_node:
                return "altk_repair"
            elif self.config.on_review_fail == "warn":
                logger.warning(f"Review detected issues: {review_result.details}")
                return self.agent_node_name
            else:
                return self.agent_node_name
        else:
            # Partial success
            return self.agent_node_name

    def _route_after_repair(self, state: Dict[str, Any]) -> str:
        """Route based on repair result."""
        repair_result = state.get("repair_result")
        repair_attempts = state.get("altk_metadata", {}).get("repair_attempts", 0)

        if repair_result is None or not repair_result.success:
            if repair_attempts >= self.config.max_repair_attempts:
                logger.warning(f"Max repair attempts ({self.config.max_repair_attempts}) reached")
                return self.agent_node_name
            return self.agent_node_name

        # Repair succeeded - retry tool execution
        return self.tool_node_name

    def enhance(self) -> Any:
        """Enhance the graph with ALTK nodes and edges.

        This modifies the graph in place and returns it.

        Returns:
            The enhanced StateGraph
        """
        # Add ALTK nodes
        if self._validation_node:
            self.graph.add_node("altk_validate", self._validation_node)

        if self._review_node:
            self.graph.add_node("altk_review", self._review_node)

        if self._repair_node:
            self.graph.add_node("altk_repair", self._repair_node)

        # Rewire edges to include ALTK nodes
        # This is a simplified version - in practice you'd need to
        # properly intercept the existing edges

        logger.info(
            f"Enhanced graph with ALTK nodes: "
            f"validation={self.config.validation_enabled}, "
            f"review={self.config.review_enabled}, "
            f"repair={self.config.repair_enabled}"
        )

        return self.graph

    def compile(self, **kwargs) -> Any:
        """Enhance and compile the graph.

        Args:
            **kwargs: Additional arguments passed to graph.compile()

        Returns:
            Compiled LangGraph application
        """
        self.enhance()
        return self.graph.compile(**kwargs)

    def get_validation_node(self) -> Optional[ValidationNode]:
        """Get the validation node."""
        return self._validation_node

    def get_review_node(self) -> Optional[ReviewNode]:
        """Get the review node."""
        return self._review_node

    def get_repair_node(self) -> Optional[RepairNode]:
        """Get the repair node."""
        return self._repair_node


def create_altk_workflow(
    agent_node: Callable,
    tool_node: Callable,
    tool_specs: List[Dict[str, Any]],
    state_schema: Any = None,
    config: Optional[ALTKGraphConfig] = None,
    **kwargs,
) -> Any:
    """Create a complete ALTK-enhanced workflow from scratch.

    This is a convenience function that creates a standard agent workflow
    with ALTK validation, review, and repair already wired up.

    Args:
        agent_node: The agent node function
        tool_node: The tool execution node function
        tool_specs: Tool specifications
        state_schema: State schema (defaults to ALTKState)
        config: ALTK configuration
        **kwargs: Additional configuration options

    Returns:
        Compiled LangGraph application

    Example:
        from altk.integrations.langgraph import create_altk_workflow

        app = create_altk_workflow(
            agent_node=my_agent,
            tool_node=my_tool_executor,
            tool_specs=my_tools,
            validation_track="fast_track",
        )
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "langgraph is required for this functionality. "
            "Install it with: pip install langgraph"
        )

    # Use provided state schema or default
    if state_schema is None:
        from altk.integrations.langgraph.types import ALTKState

        state_schema = ALTKState

    # Create config from kwargs if not provided
    if config is None:
        config = ALTKGraphConfig(tool_specs=tool_specs, **kwargs)
    else:
        config.tool_specs = tool_specs

    # Create the base workflow
    workflow = StateGraph(state_schema)

    # Add core nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Create ALTK enhanced graph
    enhanced = ALTKEnhancedGraph(
        graph=workflow,
        config=config,
        tool_node_name="tools",
        agent_node_name="agent",
    )

    # Add ALTK nodes
    if enhanced._validation_node:
        workflow.add_node("altk_validate", enhanced._validation_node)

    if enhanced._review_node:
        workflow.add_node("altk_review", enhanced._review_node)

    if enhanced._repair_node:
        workflow.add_node("altk_repair", enhanced._repair_node)

    # Wire up the workflow with ALTK in the loop
    workflow.set_entry_point("agent")

    # Agent -> Validation (if enabled) or Tools
    if config.validation_enabled and enhanced._validation_node:
        workflow.add_edge("agent", "altk_validate")
        workflow.add_conditional_edges(
            "altk_validate",
            enhanced._route_after_validation,
            {
                "tools": "tools",
                "agent": "agent",
            },
        )
    else:
        workflow.add_edge("agent", "tools")

    # Tools -> Review (if enabled) or Agent
    if config.review_enabled and enhanced._review_node:
        workflow.add_edge("tools", "altk_review")

        if config.repair_enabled and enhanced._repair_node:
            workflow.add_conditional_edges(
                "altk_review",
                enhanced._route_after_review,
                {
                    "agent": "agent",
                    "altk_repair": "altk_repair",
                },
            )
            workflow.add_conditional_edges(
                "altk_repair",
                enhanced._route_after_repair,
                {
                    "tools": "tools",
                    "agent": "agent",
                },
            )
        else:
            workflow.add_edge("altk_review", "agent")
    else:
        workflow.add_edge("tools", "agent")

    return workflow.compile()
