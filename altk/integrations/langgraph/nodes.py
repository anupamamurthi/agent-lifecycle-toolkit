"""LangGraph nodes wrapping ALTK components.

This module provides node factories and classes that wrap ALTK components
for use in LangGraph workflows.

Example:
    from altk.integrations.langgraph import (
        create_validation_node,
        create_review_node,
        create_repair_node,
    )

    workflow = StateGraph(ALTKState)
    workflow.add_node("validate", create_validation_node(
        tool_specs=my_tools,
        track="fast_track"
    ))
    workflow.add_node("review", create_review_node())
    workflow.add_node("repair", create_repair_node(docs_path="./docs"))
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Literal
from pydantic import BaseModel, Field

from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.integrations.langgraph.types import (
    ALTKState,
    ToolCallInfo,
    ValidationResult,
    ValidationIssue,
    ValidationDecision,
    ReviewResult,
    ReviewOutcome,
    RepairResult,
)

logger = logging.getLogger(__name__)


# Type alias for track options
TrackType = Literal["syntax", "fast_track", "slow_track", "spec_free", "transformations_only"]


class ValidationNode(BaseModel):
    """LangGraph node for pre-tool validation using ALTK components.

    This node validates tool calls before execution using SPARC, Refraction,
    and/or ToolGuard components.

    Attributes:
        tool_specs: List of tool specifications
        track: Validation track (syntax, fast_track, slow_track, etc.)
        llm_client: LLM client for semantic validation
        use_refraction: Whether to use Refraction for syntax validation
        use_sparc: Whether to use SPARC for semantic validation
        use_toolguard: Whether to use ToolGuard for policy enforcement
        policies_path: Path to policy files for ToolGuard
    """

    tool_specs: List[Dict[str, Any]] = Field(default_factory=list)
    track: TrackType = Field(default="fast_track")
    llm_client: Optional[Any] = Field(default=None)
    use_refraction: bool = Field(default=True)
    use_sparc: bool = Field(default=True)
    use_toolguard: bool = Field(default=False)
    policies_path: Optional[str] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    _sparc_component: Optional[Any] = None
    _refraction_component: Optional[Any] = None
    _toolguard_component: Optional[Any] = None
    _initialized: bool = False

    def _lazy_init(self):
        """Lazily initialize components on first use."""
        if self._initialized:
            return

        config = ComponentConfig(llm_client=self.llm_client) if self.llm_client else ComponentConfig()

        # Initialize SPARC if enabled
        if self.use_sparc:
            if not self.llm_client:
                logger.warning(
                    "SPARC requires a ValidatingLLMClient but no llm_client was provided. "
                    "Skipping SPARC initialization. Pass llm_client to enable semantic validation."
                )
            else:
                try:
                    from altk.pre_tool.sparc import SPARCReflectionComponent
                    from altk.pre_tool.core import Track

                    track_map = {
                        "syntax": Track.SYNTAX,
                        "fast_track": Track.FAST_TRACK,
                        "slow_track": Track.SLOW_TRACK,
                        "spec_free": Track.SPEC_FREE,
                        "transformations_only": Track.TRANSFORMATIONS_ONLY,
                    }
                    self._sparc_component = SPARCReflectionComponent(
                        config=config,
                        track=track_map.get(self.track, Track.FAST_TRACK),
                    )
                    logger.info(f"Initialized SPARC component with track: {self.track}")
                except ImportError as e:
                    logger.warning(f"Could not import SPARC component: {e}")
                except Exception as e:
                    logger.warning(f"Could not initialize SPARC component: {e}")

        # Initialize Refraction if enabled
        if self.use_refraction:
            try:
                from altk.pre_tool.refraction import RefractionComponent

                self._refraction_component = RefractionComponent(config=config)
                logger.info("Initialized Refraction component")
            except ImportError as e:
                logger.warning(f"Could not import Refraction component: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize Refraction component: {e}")

        # Initialize ToolGuard if enabled
        if self.use_toolguard and self.policies_path:
            try:
                from altk.pre_tool.toolguard import ToolGuardComponent

                self._toolguard_component = ToolGuardComponent(
                    config=config,
                    policies_path=self.policies_path,
                )
                logger.info("Initialized ToolGuard component")
            except ImportError as e:
                logger.warning(f"Could not import ToolGuard component: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize ToolGuard component: {e}")

        self._initialized = True

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation on the current tool call.

        Args:
            state: LangGraph state containing tool call information

        Returns:
            Updated state with validation_result
        """
        self._lazy_init()

        # Extract tool call from state
        tool_call = state.get("current_tool_call")
        if tool_call is None:
            # Try to extract from messages
            messages = state.get("messages", [])
            tool_call = self._extract_tool_call_from_messages(messages)

        if tool_call is None:
            return {
                "validation_result": ValidationResult(
                    decision=ValidationDecision.ERROR,
                    issues=[
                        ValidationIssue(
                            issue_type="error",
                            metric_name="input_validation",
                            explanation="No tool call found in state",
                        )
                    ],
                )
            }

        # Convert to ToolCallInfo if needed
        if isinstance(tool_call, dict):
            tool_call = ToolCallInfo(
                name=tool_call.get("name", tool_call.get("function", {}).get("name", "")),
                arguments=tool_call.get("arguments", tool_call.get("function", {}).get("arguments", {})),
                id=tool_call.get("id"),
            )

        # Get tool specs from state or use configured ones
        tool_specs = state.get("tool_specs", self.tool_specs)
        messages = state.get("messages", [])

        # Run validation
        validation_result = self._validate(tool_call, tool_specs, messages)

        return {"validation_result": validation_result}

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution of validation.

        Args:
            state: LangGraph state containing tool call information

        Returns:
            Updated state with validation_result
        """
        self._lazy_init()

        # Extract tool call from state
        tool_call = state.get("current_tool_call")
        if tool_call is None:
            messages = state.get("messages", [])
            tool_call = self._extract_tool_call_from_messages(messages)

        if tool_call is None:
            return {
                "validation_result": ValidationResult(
                    decision=ValidationDecision.ERROR,
                    issues=[
                        ValidationIssue(
                            issue_type="error",
                            metric_name="input_validation",
                            explanation="No tool call found in state",
                        )
                    ],
                )
            }

        # Convert to ToolCallInfo if needed
        if isinstance(tool_call, dict):
            tool_call = ToolCallInfo(
                name=tool_call.get("name", tool_call.get("function", {}).get("name", "")),
                arguments=tool_call.get("arguments", tool_call.get("function", {}).get("arguments", {})),
                id=tool_call.get("id"),
            )

        tool_specs = state.get("tool_specs", self.tool_specs)
        messages = state.get("messages", [])

        validation_result = await self._avalidate(tool_call, tool_specs, messages)

        return {"validation_result": validation_result}

    def _extract_tool_call_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract the most recent tool call from messages."""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    return tool_calls[-1]
        return None

    def _validate(
        self,
        tool_call: ToolCallInfo,
        tool_specs: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Run validation using configured components."""
        issues = []
        decision = ValidationDecision.APPROVE
        execution_time_ms = 0.0
        corrected_tool_call = None

        # Prepare input for SPARC
        if self._sparc_component:
            try:
                from altk.pre_tool.core import SPARCReflectionRunInput

                sparc_input = SPARCReflectionRunInput(
                    messages=messages,
                    tool_specs=tool_specs,
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                    ],
                )

                sparc_output = self._sparc_component.process(
                    sparc_input, phase=AgentPhase.RUNTIME
                )

                if sparc_output.output:
                    result = sparc_output.output.reflection_result
                    execution_time_ms = sparc_output.output.execution_time_ms

                    # Map SPARC decision to our decision
                    if result.decision.value == "reject":
                        decision = ValidationDecision.REJECT
                    elif result.decision.value == "error":
                        decision = ValidationDecision.ERROR

                    # Map issues
                    for issue in result.issues:
                        issues.append(
                            ValidationIssue(
                                issue_type=issue.issue_type.value,
                                metric_name=issue.metric_name,
                                explanation=issue.explanation,
                                correction=issue.correction,
                            )
                        )

                        # Extract correction if available
                        if issue.correction and not corrected_tool_call:
                            corrected_args = {**tool_call.arguments}
                            corrected_args.update(issue.correction)
                            corrected_tool_call = ToolCallInfo(
                                name=tool_call.name,
                                arguments=corrected_args,
                                id=tool_call.id,
                            )

            except Exception as e:
                logger.error(f"SPARC validation failed: {e}")
                issues.append(
                    ValidationIssue(
                        issue_type="error",
                        metric_name="sparc",
                        explanation=f"SPARC validation error: {str(e)}",
                    )
                )

        return ValidationResult(
            decision=decision,
            issues=issues,
            execution_time_ms=execution_time_ms,
            original_tool_call=tool_call,
            corrected_tool_call=corrected_tool_call,
        )

    async def _avalidate(
        self,
        tool_call: ToolCallInfo,
        tool_specs: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Run async validation using configured components."""
        issues = []
        decision = ValidationDecision.APPROVE
        execution_time_ms = 0.0
        corrected_tool_call = None

        if self._sparc_component:
            try:
                from altk.pre_tool.core import SPARCReflectionRunInput

                sparc_input = SPARCReflectionRunInput(
                    messages=messages,
                    tool_specs=tool_specs,
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                    ],
                )

                sparc_output = await self._sparc_component.aprocess(
                    sparc_input, phase=AgentPhase.RUNTIME
                )

                if sparc_output.output:
                    result = sparc_output.output.reflection_result
                    execution_time_ms = sparc_output.output.execution_time_ms

                    if result.decision.value == "reject":
                        decision = ValidationDecision.REJECT
                    elif result.decision.value == "error":
                        decision = ValidationDecision.ERROR

                    for issue in result.issues:
                        issues.append(
                            ValidationIssue(
                                issue_type=issue.issue_type.value,
                                metric_name=issue.metric_name,
                                explanation=issue.explanation,
                                correction=issue.correction,
                            )
                        )

                        if issue.correction and not corrected_tool_call:
                            corrected_args = {**tool_call.arguments}
                            corrected_args.update(issue.correction)
                            corrected_tool_call = ToolCallInfo(
                                name=tool_call.name,
                                arguments=corrected_args,
                                id=tool_call.id,
                            )

            except Exception as e:
                logger.error(f"SPARC async validation failed: {e}")
                issues.append(
                    ValidationIssue(
                        issue_type="error",
                        metric_name="sparc",
                        explanation=f"SPARC validation error: {str(e)}",
                    )
                )

        return ValidationResult(
            decision=decision,
            issues=issues,
            execution_time_ms=execution_time_ms,
            original_tool_call=tool_call,
            corrected_tool_call=corrected_tool_call,
        )


class ReviewNode(BaseModel):
    """LangGraph node for post-tool review using ALTK SilentReview.

    This node reviews tool outputs to detect silent errors - cases where
    the tool executes successfully but returns incorrect or incomplete data.

    Attributes:
        llm_client: LLM client for review
        review_type: Type of review (json or tabular)
    """

    llm_client: Optional[Any] = Field(default=None)
    review_type: Literal["json", "tabular"] = Field(default="json")

    model_config = {"arbitrary_types_allowed": True}

    _review_component: Optional[Any] = None
    _initialized: bool = False

    def _lazy_init(self):
        """Lazily initialize the review component."""
        if self._initialized:
            return

        if not self.llm_client:
            logger.warning(
                "SilentReview requires an LLM client but no llm_client was provided. "
                "Skipping review initialization. Pass llm_client or set LLM_PROVIDER to enable review."
            )
            self._initialized = True
            return

        config = ComponentConfig(llm_client=self.llm_client)

        try:
            if self.review_type == "tabular":
                from altk.post_tool.silent_review.silent_review import SilentReviewForTabularDataComponent

                self._review_component = SilentReviewForTabularDataComponent(config=config)
            else:
                from altk.post_tool.silent_review.silent_review import SilentReviewForJSONDataComponent

                self._review_component = SilentReviewForJSONDataComponent(config=config)

            logger.info(f"Initialized SilentReview component (type: {self.review_type})")
        except ImportError as e:
            logger.warning(f"Could not import SilentReview component: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize SilentReview component: {e}")

        self._initialized = True

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute review on the tool response.

        Args:
            state: LangGraph state containing tool response

        Returns:
            Updated state with review_result
        """
        self._lazy_init()

        tool_response = state.get("tool_response")
        messages = state.get("messages", [])
        tool_specs = state.get("tool_specs", [])

        if tool_response is None:
            return {
                "review_result": ReviewResult(
                    outcome=ReviewOutcome.NOT_ACCOMPLISHED,
                    details={"error": "No tool response found in state"},
                )
            }

        review_result = self._review(tool_response, messages, tool_specs)
        return {"review_result": review_result}

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution of review.

        Args:
            state: LangGraph state containing tool response

        Returns:
            Updated state with review_result
        """
        self._lazy_init()

        tool_response = state.get("tool_response")
        messages = state.get("messages", [])
        tool_specs = state.get("tool_specs", [])

        if tool_response is None:
            return {
                "review_result": ReviewResult(
                    outcome=ReviewOutcome.NOT_ACCOMPLISHED,
                    details={"error": "No tool response found in state"},
                )
            }

        review_result = await self._areview(tool_response, messages, tool_specs)
        return {"review_result": review_result}

    def _review(
        self,
        tool_response: Any,
        messages: List[Dict[str, Any]],
        tool_specs: List[Dict[str, Any]],
    ) -> ReviewResult:
        """Run review using the configured component."""
        if not self._review_component:
            return ReviewResult(
                outcome=ReviewOutcome.ACCOMPLISHED,
                details={"warning": "Review component not available"},
                tool_response=tool_response,
            )

        try:
            from altk.post_tool.core.toolkit import SilentReviewRunInput, Outcome

            # Get the first tool spec if available
            tool_spec = tool_specs[0] if tool_specs else None

            review_input = SilentReviewRunInput(
                messages=messages,
                tool_response=tool_response,
                tool_spec=tool_spec,
            )

            review_output = self._review_component.process(
                review_input, phase=AgentPhase.RUNTIME
            )

            # Map Outcome to ReviewOutcome
            outcome_map = {
                Outcome.ACCOMPLISHED: ReviewOutcome.ACCOMPLISHED,
                Outcome.PARTIAL_ACCOMPLISH: ReviewOutcome.PARTIAL,
                Outcome.NOT_ACCOMPLISHED: ReviewOutcome.NOT_ACCOMPLISHED,
            }

            return ReviewResult(
                outcome=outcome_map.get(review_output.outcome, ReviewOutcome.NOT_ACCOMPLISHED),
                details=review_output.details,
                tool_response=tool_response,
            )

        except Exception as e:
            logger.error(f"Review failed: {e}")
            return ReviewResult(
                outcome=ReviewOutcome.NOT_ACCOMPLISHED,
                details={"error": str(e)},
                tool_response=tool_response,
            )

    async def _areview(
        self,
        tool_response: Any,
        messages: List[Dict[str, Any]],
        tool_specs: List[Dict[str, Any]],
    ) -> ReviewResult:
        """Run async review using the configured component."""
        if not self._review_component:
            return ReviewResult(
                outcome=ReviewOutcome.ACCOMPLISHED,
                details={"warning": "Review component not available"},
                tool_response=tool_response,
            )

        try:
            from altk.post_tool.core.toolkit import SilentReviewRunInput, Outcome

            tool_spec = tool_specs[0] if tool_specs else None

            review_input = SilentReviewRunInput(
                messages=messages,
                tool_response=tool_response,
                tool_spec=tool_spec,
            )

            review_output = await self._review_component.aprocess(
                review_input, phase=AgentPhase.RUNTIME
            )

            outcome_map = {
                Outcome.ACCOMPLISHED: ReviewOutcome.ACCOMPLISHED,
                Outcome.PARTIAL_ACCOMPLISH: ReviewOutcome.PARTIAL,
                Outcome.NOT_ACCOMPLISHED: ReviewOutcome.NOT_ACCOMPLISHED,
            }

            return ReviewResult(
                outcome=outcome_map.get(review_output.outcome, ReviewOutcome.NOT_ACCOMPLISHED),
                details=review_output.details,
                tool_response=tool_response,
            )

        except Exception as e:
            logger.error(f"Async review failed: {e}")
            return ReviewResult(
                outcome=ReviewOutcome.NOT_ACCOMPLISHED,
                details={"error": str(e)},
                tool_response=tool_response,
            )


class RepairNode(BaseModel):
    """LangGraph node for tool call repair using ALTK RAGRepair.

    This node attempts to repair failed tool calls using RAG-based
    suggestions from domain documentation.

    Attributes:
        docs_path: Path to documentation for RAG
        llm_client: LLM client for repair suggestions
        retrieval_type: Type of retrieval (bm25 or chromadb)
    """

    docs_path: str = Field(description="Path to documentation")
    llm_client: Optional[Any] = Field(default=None)
    retrieval_type: Literal["bm25", "chromadb"] = Field(default="chromadb")

    model_config = {"arbitrary_types_allowed": True}

    _repair_component: Optional[Any] = None
    _initialized: bool = False

    def _lazy_init(self):
        """Lazily initialize the repair component."""
        if self._initialized:
            return

        try:
            from altk.post_tool.rag_repair import RAGRepairComponent
            from altk.post_tool.rag_repair.rag_repair_config import RAGRepairComponentConfig

            config = RAGRepairComponentConfig(
                llm_client=self.llm_client,
                retrieval_type=self.retrieval_type,
            )

            self._repair_component = RAGRepairComponent(
                docs_path=self.docs_path,
                config=config,
            )

            # Setup RAG (build phase)
            self._repair_component.setup_rag()
            logger.info(f"Initialized RAGRepair component with docs from: {self.docs_path}")

        except ImportError as e:
            logger.warning(f"Could not import RAGRepair component: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize RAGRepair component: {e}")

        self._initialized = True

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repair on the failed tool call.

        Args:
            state: LangGraph state containing error information

        Returns:
            Updated state with repair_result
        """
        self._lazy_init()

        # Extract information needed for repair
        current_tool_call = state.get("current_tool_call")
        messages = state.get("messages", [])
        error = state.get("tool_error") or state.get("error")

        if current_tool_call is None:
            return {
                "repair_result": RepairResult(
                    success=False,
                    original_tool_call="",
                    repaired_tool_call=None,
                )
            }

        # Convert tool call to string representation
        if isinstance(current_tool_call, ToolCallInfo):
            tool_call_str = f"{current_tool_call.name}({current_tool_call.arguments})"
        elif isinstance(current_tool_call, dict):
            name = current_tool_call.get("name", "")
            args = current_tool_call.get("arguments", {})
            tool_call_str = f"{name}({args})"
        else:
            tool_call_str = str(current_tool_call)

        repair_result = self._repair(tool_call_str, messages, error)
        return {"repair_result": repair_result}

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution of repair.

        Args:
            state: LangGraph state containing error information

        Returns:
            Updated state with repair_result
        """
        # RAGRepair doesn't have async implementation, so we call sync
        return self.__call__(state)

    def _repair(
        self,
        tool_call: str,
        messages: List[Dict[str, Any]],
        error: Optional[str],
    ) -> RepairResult:
        """Run repair using the configured component."""
        if not self._repair_component:
            return RepairResult(
                success=False,
                original_tool_call=tool_call,
                repaired_tool_call=None,
            )

        try:
            from altk.post_tool.core.toolkit import RAGRepairRunInput

            repair_input = RAGRepairRunInput(
                messages=messages,
                tool_call=tool_call,
                error=error,
            )

            repair_output = self._repair_component.process(
                repair_input, phase=AgentPhase.RUNTIME
            )

            return RepairResult(
                success=repair_output.new_tool_call != tool_call,
                original_tool_call=tool_call,
                repaired_tool_call=repair_output.new_tool_call,
                repair_result=repair_output.result,
                retrieved_docs=repair_output.retrieved_docs,
            )

        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return RepairResult(
                success=False,
                original_tool_call=tool_call,
                repaired_tool_call=None,
            )


# Factory functions for creating nodes


def create_validation_node(
    tool_specs: Optional[List[Dict[str, Any]]] = None,
    track: TrackType = "fast_track",
    llm_client: Optional[Any] = None,
    use_refraction: bool = True,
    use_sparc: Optional[bool] = None,
    use_toolguard: bool = False,
    policies_path: Optional[str] = None,
) -> ValidationNode:
    """Create a validation node for LangGraph.

    Args:
        tool_specs: List of tool specifications
        track: Validation track (syntax, fast_track, slow_track, etc.)
        llm_client: LLM client for semantic validation
        use_refraction: Whether to use Refraction for syntax validation
        use_sparc: Whether to use SPARC for semantic validation.
            If None (default), auto-detected based on track:
            - "syntax": SPARC disabled (no LLM needed)
            - other tracks: SPARC enabled if llm_client provided
        use_toolguard: Whether to use ToolGuard for policy enforcement
        policies_path: Path to policy files for ToolGuard

    Returns:
        Configured ValidationNode

    Example:
        workflow.add_node("validate", create_validation_node(
            tool_specs=my_tools,
            track="fast_track"
        ))
    """
    # Auto-detect use_sparc based on track and llm_client
    if use_sparc is None:
        # Syntax track doesn't need SPARC (no LLM required)
        # Other tracks enable SPARC only if llm_client is provided
        use_sparc = track != "syntax" and llm_client is not None

    return ValidationNode(
        tool_specs=tool_specs or [],
        track=track,
        llm_client=llm_client,
        use_refraction=use_refraction,
        use_sparc=use_sparc,
        use_toolguard=use_toolguard,
        policies_path=policies_path,
    )


def create_review_node(
    llm_client: Optional[Any] = None,
    review_type: Literal["json", "tabular"] = "json",
) -> ReviewNode:
    """Create a review node for LangGraph.

    Args:
        llm_client: LLM client for review
        review_type: Type of review (json or tabular)

    Returns:
        Configured ReviewNode

    Example:
        workflow.add_node("review", create_review_node())
    """
    return ReviewNode(
        llm_client=llm_client,
        review_type=review_type,
    )


def create_repair_node(
    docs_path: str,
    llm_client: Optional[Any] = None,
    retrieval_type: Literal["bm25", "chromadb"] = "chromadb",
) -> RepairNode:
    """Create a repair node for LangGraph.

    Args:
        docs_path: Path to documentation for RAG
        llm_client: LLM client for repair suggestions
        retrieval_type: Type of retrieval (bm25 or chromadb)

    Returns:
        Configured RepairNode

    Example:
        workflow.add_node("repair", create_repair_node(docs_path="./docs"))
    """
    return RepairNode(
        docs_path=docs_path,
        llm_client=llm_client,
        retrieval_type=retrieval_type,
    )
