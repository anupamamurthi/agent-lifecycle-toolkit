# ALTK LangGraph Integration

This module provides seamless integration between ALTK (Agent Lifecycle Toolkit) and [LangGraph](https://github.com/langchain-ai/langgraph), enabling you to add validation, review, and repair capabilities to your LangGraph agents.

## Installation

```bash
# Install ALTK with LangGraph support
pip install altk langgraph langchain-core
```

## Quick Start

### Option 1: Prebuilt Agents (Easiest)

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from altk.integrations.langgraph import create_altk_react_agent

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# Create agent with ALTK validation built-in
agent = create_altk_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[get_weather, search_web],
    validation_track="fast_track",
)

# Use it
result = agent.invoke({
    "messages": [("user", "What's the weather in San Francisco?")]
})
print(result["messages"][-1].content)
```

### Option 2: Manual Node Wiring (Full Control)

```python
from langgraph.graph import StateGraph, END
from altk.integrations.langgraph import (
    create_validation_node,
    create_review_node,
    ValidationDecision,
)

# Create ALTK nodes
validation_node = create_validation_node(
    tool_specs=my_tools,
    track="fast_track",
)
review_node = create_review_node(review_type="json")

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("validate", validation_node)
workflow.add_node("tools", tool_node)
workflow.add_node("review", review_node)

# Wire edges
workflow.set_entry_point("agent")
workflow.add_edge("agent", "validate")
workflow.add_conditional_edges("validate", route_on_validation)
workflow.add_edge("tools", "review")
workflow.add_edge("review", "agent")

app = workflow.compile()
```

### Option 2: Enhanced Graph Wrapper (Quick Enhancement)

```python
from altk.integrations.langgraph import ALTKEnhancedGraph, ALTKGraphConfig

# Your existing workflow
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

app = enhanced.compile()
```

### Option 3: One-Liner (Standard Patterns)

```python
from altk.integrations.langgraph import create_altk_workflow

app = create_altk_workflow(
    agent_node=my_agent,
    tool_node=my_tool_executor,
    tool_specs=my_tools,
    validation_track="fast_track",
)
```

## Try It Out

### Run the Example

```bash
cd /Users/anu/agent-lifecycle-toolkit

# Install dependencies
pip install langgraph langchain-core

# Run the example
python examples/try_langgraph_altk.py
```

### What It Does

The example creates a workflow with ALTK validation and review:

```
User Query → Agent → Validate → Tools → Review → Agent → END
                ↑                           |
                └───────────────────────────┘
                    (if validation fails)
```

1. **Agent** decides to call `get_weather` tool
2. **ValidationNode** checks the tool call (syntax/semantic validation)
3. **Tool** executes and returns weather data
4. **ReviewNode** checks if the response looks correct
5. **Agent** sees the results and continues

### Expected Output

```
============================================================
ALTK LangGraph Integration Demo
============================================================

1. Creating ALTK nodes...
   - ValidationNode created (syntax track)
   - ReviewNode created (JSON review)

2. Building LangGraph workflow...
   - Workflow compiled successfully!

3. Running workflow...
------------------------------------------------------------
  [Agent] Deciding to call get_weather tool...
  [Router] Validation decision: approve
  [Tool] Executing get_weather with args: {'city': 'San Francisco', 'units': 'celsius'}
  [Tool] Response: {'temperature': 18, 'units': 'celsius', ...}
  [Router] Review outcome: accomplished
------------------------------------------------------------

4. Final Results:
   - Validation: approve
   - Review: accomplished
   - Tool response: {'temperature': 18, ...}
```

### Try Breaking It

Edit the example to test validation rejection:

```python
# Change the tool call to have invalid arguments
"current_tool_call": {
    "name": "search_database",
    "arguments": {"limit": 500},  # Missing 'query', limit > 100
},
```

## Prebuilt Agents

ALTK provides ready-to-use agents with validation, review, and repair built-in.

### create_altk_react_agent

Full-featured ReAct agent with all ALTK capabilities:

```python
from altk.integrations.langgraph import create_altk_react_agent

agent = create_altk_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[tool1, tool2],

    # ALTK options
    validation_enabled=True,
    review_enabled=True,
    repair_enabled=True,
    validation_track="fast_track",
    docs_path="./docs/",           # For RAG repair
    policies_path="./policies/",   # For ToolGuard

    # Behavior
    on_validation_fail="reject",   # reject, warn, ignore
    on_review_fail="repair",       # repair, warn, ignore
    max_repair_attempts=3,

    # Standard LangGraph options
    checkpointer=memory,
    messages_modifier="You are a helpful assistant.",
)
```

### create_altk_safe_agent

Maximum safety with all protections enabled:

```python
from altk.integrations.langgraph import create_altk_safe_agent

# For high-stakes applications
agent = create_altk_safe_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[database_tool, api_tool],
    policies_path="./policies/",   # Policy enforcement
    docs_path="./docs/",           # RAG repair
)
```

Features:
- Full semantic validation (`slow_track`)
- ToolGuard policy enforcement
- Silent error detection
- Automatic RAG repair

### create_altk_fast_agent

Minimal overhead for speed-critical applications:

```python
from altk.integrations.langgraph import create_altk_fast_agent

# For speed-critical applications
agent = create_altk_fast_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[quick_tool],
)
```

Features:
- Syntax-only validation (no LLM calls)
- No review step
- Minimal latency overhead

### create_altk_tool_agent

Simple tool agent with basic validation:

```python
from altk.integrations.langgraph import create_altk_tool_agent

agent = create_altk_tool_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[calculator, search],
    validation_track="syntax",
    review_enabled=True,
)
```

### Comparison

| Agent | Validation | Review | Repair | Use Case |
|-------|------------|--------|--------|----------|
| `create_altk_react_agent` | Configurable | Configurable | Configurable | General purpose |
| `create_altk_safe_agent` | Full (slow_track) | Yes | Yes | High-stakes, enterprise |
| `create_altk_fast_agent` | Syntax only | No | No | Speed-critical |
| `create_altk_tool_agent` | Syntax | Yes | No | Simple tool calling |

## Components

### ValidationNode

Validates tool calls before execution using ALTK's pre-tool components.

```python
from altk.integrations.langgraph import create_validation_node

node = create_validation_node(
    tool_specs=my_tools,           # Tool specifications
    track="fast_track",            # Validation track (syntax, fast_track, slow_track)
    llm_client=my_llm,             # LLM for semantic validation
    use_sparc=True,                # Enable SPARC validation
    use_refraction=True,           # Enable Refraction syntax checking
    use_toolguard=False,           # Enable ToolGuard policy enforcement
    policies_path="./policies/",   # Path to policy files
)
```

**Validation Tracks:**

| Track | LLM Calls | Description |
|-------|-----------|-------------|
| `syntax` | 0 | Fast static validation only |
| `fast_track` | 2 | General hallucination + function selection |
| `slow_track` | 5+ | Comprehensive with transformations |
| `spec_free` | 1 | Semantic validation without specs |

### ReviewNode

Reviews tool outputs to detect silent errors.

```python
from altk.integrations.langgraph import create_review_node

node = create_review_node(
    llm_client=my_llm,      # LLM for review
    review_type="json",     # "json" or "tabular"
)
```

### RepairNode

Repairs failed tool calls using RAG-based suggestions.

```python
from altk.integrations.langgraph import create_repair_node

node = create_repair_node(
    docs_path="./docs/",         # Path to documentation
    llm_client=my_llm,           # LLM for repair
    retrieval_type="chromadb",   # "bm25" or "chromadb"
)
```

## State Types

### ALTKState

Full state type with all ALTK fields:

```python
from altk.integrations.langgraph import ALTKState

class MyState(ALTKState):
    # Inherits:
    # - messages: List[Dict]
    # - tool_specs: List[Dict]
    # - current_tool_call: Optional[ToolCallInfo]
    # - tool_response: Optional[Any]
    # - validation_result: Optional[ValidationResult]
    # - review_result: Optional[ReviewResult]
    # - repair_result: Optional[RepairResult]

    # Add your custom fields
    my_custom_field: str
```

### Result Types

```python
from altk.integrations.langgraph import (
    ValidationResult,
    ValidationDecision,  # APPROVE, REJECT, ERROR
    ReviewResult,
    ReviewOutcome,       # ACCOMPLISHED, PARTIAL, NOT_ACCOMPLISHED
    RepairResult,
)

# Check validation result
if validation_result.should_proceed:
    execute_tool()

if validation_result.has_corrections:
    use_corrected_call(validation_result.corrected_tool_call)

# Check review result
if review_result.needs_repair:
    trigger_repair()
```

## Configuration

### ALTKGraphConfig

Full configuration options:

```python
from altk.integrations.langgraph import ALTKGraphConfig

config = ALTKGraphConfig(
    # Feature toggles
    validation_enabled=True,
    review_enabled=True,
    repair_enabled=False,

    # Tool configuration
    tool_specs=my_tools,

    # Validation settings
    validation_track="fast_track",
    use_toolguard=False,
    policies_path=None,

    # Review settings
    review_type="json",  # or "tabular"

    # Repair settings
    docs_path="./docs/",
    retrieval_type="chromadb",
    max_repair_attempts=3,

    # LLM client (shared across components)
    llm_client=my_llm,

    # Behavior on failures
    on_validation_fail="reject",  # "reject", "warn", "ignore"
    on_review_fail="repair",      # "repair", "warn", "ignore"
)
```

## Workflow Patterns

### Pattern 1: Validation Gate

Only execute tools that pass validation:

```python
workflow.add_conditional_edges(
    "validate",
    lambda s: "tools" if s["validation_result"].should_proceed else "agent",
)
```

### Pattern 2: Review with Retry

Retry tools that fail review:

```python
def route_after_review(state):
    if state["review_result"].needs_repair:
        return "repair"
    return "agent"

workflow.add_conditional_edges("review", route_after_review)
workflow.add_edge("repair", "tools")  # Retry with repaired call
```

### Pattern 3: Validation with Correction

Use corrected tool calls when available:

```python
def tool_node(state):
    validation = state.get("validation_result")

    # Use corrected call if available
    if validation and validation.has_corrections:
        tool_call = validation.corrected_tool_call
    else:
        tool_call = state["current_tool_call"]

    return execute_tool(tool_call)
```

## API Reference

### Node Factories

| Function | Description |
|----------|-------------|
| `create_validation_node()` | Create a pre-tool validation node |
| `create_review_node()` | Create a post-tool review node |
| `create_repair_node()` | Create a tool repair node |

### Classes

| Class | Description |
|-------|-------------|
| `ValidationNode` | LangGraph node for SPARC/Refraction/ToolGuard |
| `ReviewNode` | LangGraph node for Silent Review |
| `RepairNode` | LangGraph node for RAG Repair |
| `ALTKEnhancedGraph` | Wrapper to enhance existing graphs |
| `ALTKGraphConfig` | Configuration for enhanced graphs |

### Types

| Type | Description |
|------|-------------|
| `ALTKState` | Full state TypedDict with ALTK fields |
| `ToolCallInfo` | Tool call information model |
| `ValidationResult` | Result from validation node |
| `ReviewResult` | Result from review node |
| `RepairResult` | Result from repair node |
| `ValidationDecision` | Enum: APPROVE, REJECT, ERROR |
| `ReviewOutcome` | Enum: ACCOMPLISHED, PARTIAL, NOT_ACCOMPLISHED |

## Examples

See the [examples directory](../../../examples/) for complete working examples:

- `try_langgraph_altk.py` - Simple runnable demo
- `langgraph_altk_integration_example.py` - Comprehensive examples

## Troubleshooting

### "No validation result"

Ensure your state includes `current_tool_call` or the messages contain tool calls:

```python
state = {
    "current_tool_call": {
        "name": "get_weather",
        "arguments": {"city": "NYC"},
    },
    # ... or messages with tool_calls
}
```

### "SPARC component not initialized"

SPARC requires a `ValidatingLLMClient`. For syntax-only validation, use:

```python
create_validation_node(track="syntax")  # No LLM needed
```

### Import errors

Install all dependencies:

```bash
pip install altk[langgraph]  # When available
# Or manually:
pip install langgraph langchain-core
```
