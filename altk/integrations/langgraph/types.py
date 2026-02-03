"""Type definitions for ALTK LangGraph integration.

This module defines the state types and data structures used by ALTK
nodes in LangGraph workflows.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from enum import Enum
from pydantic import BaseModel, Field
import operator


class ValidationDecision(str, Enum):
    """Decision from validation node."""

    APPROVE = "approve"
    REJECT = "reject"
    ERROR = "error"


class ReviewOutcome(str, Enum):
    """Outcome from review node."""

    ACCOMPLISHED = "accomplished"
    PARTIAL = "partial"
    NOT_ACCOMPLISHED = "not_accomplished"


class ToolCallInfo(BaseModel):
    """Information about a tool call."""

    name: str = Field(description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool"
    )
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this tool call"
    )


class ValidationIssue(BaseModel):
    """An issue identified during validation."""

    issue_type: str = Field(description="Type of issue (static, semantic, etc.)")
    metric_name: str = Field(description="Name of the metric that identified this issue")
    explanation: str = Field(description="Human-readable explanation of the issue")
    correction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Suggested correction if available"
    )


class ValidationResult(BaseModel):
    """Result from validation node."""

    decision: ValidationDecision = Field(
        description="Whether the tool call should proceed"
    )
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="List of issues found during validation"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Time taken for validation in milliseconds"
    )
    original_tool_call: Optional[ToolCallInfo] = Field(
        default=None,
        description="The original tool call that was validated"
    )
    corrected_tool_call: Optional[ToolCallInfo] = Field(
        default=None,
        description="Corrected tool call if corrections were suggested"
    )

    @property
    def should_proceed(self) -> bool:
        """Check if the tool call should proceed."""
        return self.decision == ValidationDecision.APPROVE

    @property
    def has_corrections(self) -> bool:
        """Check if corrections are available."""
        return self.corrected_tool_call is not None


class ReviewResult(BaseModel):
    """Result from review node."""

    outcome: ReviewOutcome = Field(
        description="Assessment of tool output quality"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed review information"
    )
    tool_response: Optional[Any] = Field(
        default=None,
        description="The tool response that was reviewed"
    )

    @property
    def needs_repair(self) -> bool:
        """Check if the tool output needs repair."""
        return self.outcome == ReviewOutcome.NOT_ACCOMPLISHED

    @property
    def is_partial(self) -> bool:
        """Check if the tool output was partially successful."""
        return self.outcome == ReviewOutcome.PARTIAL


class RepairResult(BaseModel):
    """Result from repair node."""

    success: bool = Field(
        description="Whether repair was successful"
    )
    original_tool_call: str = Field(
        description="The original tool call that failed"
    )
    repaired_tool_call: Optional[str] = Field(
        default=None,
        description="The repaired tool call"
    )
    repair_result: Optional[Any] = Field(
        default=None,
        description="Result from executing the repaired tool call"
    )
    retrieved_docs: str = Field(
        default="",
        description="Documents retrieved to help with repair"
    )


# Reducer function for appending to lists
def add_messages(left: List, right: List) -> List:
    """Reducer that appends messages."""
    return left + right


class ALTKState(TypedDict, total=False):
    """State for ALTK-enhanced LangGraph workflows.

    This state extends the typical LangGraph agent state with ALTK-specific
    fields for tracking validation, review, and repair results.

    Attributes:
        messages: Conversation messages (uses add_messages reducer)
        tool_specs: Available tool specifications
        current_tool_call: The tool call being processed
        tool_response: Response from tool execution

        # ALTK-specific fields
        validation_result: Result from validation node
        review_result: Result from review node
        repair_result: Result from repair node
        altk_metadata: Additional metadata for ALTK processing
    """

    # Standard agent state
    messages: Annotated[List[Dict[str, Any]], operator.add]
    tool_specs: List[Dict[str, Any]]
    current_tool_call: Optional[ToolCallInfo]
    tool_response: Optional[Any]

    # ALTK state
    validation_result: Optional[ValidationResult]
    review_result: Optional[ReviewResult]
    repair_result: Optional[RepairResult]
    altk_metadata: Dict[str, Any]


class MinimalALTKState(TypedDict, total=False):
    """Minimal state for simple ALTK workflows.

    Use this when you want to add ALTK to an existing workflow
    without changing your state structure significantly.
    """

    messages: Annotated[List[Dict[str, Any]], operator.add]
    altk_validation: Optional[ValidationResult]
    altk_review: Optional[ReviewResult]
    altk_repair: Optional[RepairResult]
