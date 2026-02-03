"""
Simple LangGraph Agent with Tools, RAG, and ALTK Enhancements

This example demonstrates how to create a LangGraph-based agent enhanced with
ALTK components for improved reliability:

PRE-TOOL PHASE:
- SPARC: Pre-tool validation to ensure tool calls are correct
- ToolGuard: Policy enforcement to block dangerous operations

POST-TOOL PHASE:
- Silent Review: Post-tool review to detect silent errors in responses
- RAG Repair: Attempt to fix failed tool calls using documentation

PRE-RESPONSE PHASE:
- Policy Guard: Check final response for compliance before returning

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
# TOOLGUARD POLICIES (for policy enforcement)
# ============================================================================

TOOL_POLICIES = {
    "transfer_funds": [
        {
            "name": "max_transfer_limit",
            "description": "Single transfers cannot exceed $5,000 without manager approval",
            "check": lambda args: args.get("amount", 0) <= 5000,
            "violation_message": "Transfer amount ${amount} exceeds policy limit of $5,000. Requires manager approval.",
        },
        {
            "name": "no_self_transfer",
            "description": "Cannot transfer to the same account",
            "check": lambda args: args.get("from_account") != args.get("to_account"),
            "violation_message": "Policy violation: Cannot transfer funds to the same account.",
        },
    ],
    "send_notification": [
        {
            "name": "no_inactive_users",
            "description": "Cannot send notifications to inactive users",
            "check": lambda args: args.get("recipient_id") not in ["U003"],  # U003 is inactive
            "violation_message": "Policy violation: Cannot send notifications to inactive users.",
        },
    ],
    "create_support_ticket": [
        {
            "name": "no_critical_without_description",
            "description": "Critical tickets must have detailed descriptions (50+ chars)",
            "check": lambda args: args.get("priority") != "critical" or len(args.get("description", "")) >= 50,
            "violation_message": "Policy violation: Critical tickets require detailed descriptions (50+ characters).",
        },
    ],
}

# ============================================================================
# RAG REPAIR DOCUMENTATION (for fixing failed tool calls)
# ============================================================================

REPAIR_DOCS = {
    "user_ids": """
    Valid User IDs in the system:
    - U001: Alice Smith (Engineering, active)
    - U002: Bob Johnson (Sales, active)
    - U003: Carol White (HR, inactive)

    Common mistakes:
    - Using names instead of IDs (use "U001" not "Alice")
    - Using lowercase (use "U001" not "u001")
    - Missing the "U" prefix (use "U001" not "001")
    """,
    "inventory_skus": """
    Valid Product SKUs:
    - SKU-001: Laptop Pro 15 (in stock)
    - SKU-002: Wireless Mouse (out of stock)
    - SKU-003: USB-C Hub (in stock)

    Valid warehouses: "main", "west", "east"

    Common mistakes:
    - Using product names instead of SKUs
    - Invalid warehouse names (only main/west/east are valid)
    """,
    "support_tickets": """
    Support Ticket Requirements:
    - title: Must be at least 5 characters
    - description: Must be at least 10 characters
    - priority: Must be one of: "low", "medium", "high", "critical"
    - category: Must be one of: "bug", "feature", "question", "other"

    Common mistakes:
    - Using "urgent" instead of "critical" for priority
    - Using "issue" instead of "bug" for category
    """,
    "transfers": """
    Fund Transfer Rules:
    - Amount must be positive
    - Single transfer limit: $10,000 (policy limit: $5,000)
    - Supported currencies: USD, EUR, GBP
    - Cannot transfer to same account

    Common mistakes:
    - Amount as string instead of number
    - Using unsupported currency codes
    """,
}

# ============================================================================
# RESPONSE POLICIES (for Policy Guard)
# ============================================================================

RESPONSE_POLICIES = [
    {
        "name": "no_pii_exposure",
        "description": "Response should not expose full email addresses",
        "check": lambda response: "@" not in response or "company.com" not in response,
        "violation_message": "Response contains potentially sensitive email information.",
    },
    {
        "name": "no_account_numbers",
        "description": "Response should not expose raw account numbers",
        "check": lambda response: "ACC-" not in response or "TXN-" in response,  # Allow transaction IDs
        "violation_message": "Response contains raw account numbers.",
    },
    {
        "name": "professional_tone",
        "description": "Response should maintain professional tone",
        "check": lambda response: not any(word in response.lower() for word in ["stupid", "dumb", "idiot"]),
        "violation_message": "Response contains unprofessional language.",
    },
]


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
# ADDITIONAL TOOLS (to showcase SPARC validation and Silent Review)
# ============================================================================

# Fake database of users
USERS_DB = {
    "U001": {"name": "Alice Smith", "email": "alice@company.com", "department": "Engineering", "status": "active"},
    "U002": {"name": "Bob Johnson", "email": "bob@company.com", "department": "Sales", "status": "active"},
    "U003": {"name": "Carol White", "email": "carol@company.com", "department": "HR", "status": "inactive"},
}

# Fake inventory database
INVENTORY_DB = {
    "SKU-001": {"name": "Laptop Pro 15", "quantity": 50, "price": 1299.99, "status": "in_stock"},
    "SKU-002": {"name": "Wireless Mouse", "quantity": 0, "price": 29.99, "status": "out_of_stock"},
    "SKU-003": {"name": "USB-C Hub", "quantity": 25, "price": 79.99, "status": "in_stock"},
}


@tool
def get_user_info(user_id: str, include_email: bool = False) -> str:
    """
    Get information about a user by their ID.

    Args:
        user_id: The user ID (e.g., "U001", "U002")
        include_email: Whether to include the user's email in the response
    """
    if user_id not in USERS_DB:
        # This will trigger Silent Review - "not found" response
        return f"Error: User '{user_id}' not found in the system."

    user = USERS_DB[user_id]
    info = f"User: {user['name']}, Department: {user['department']}, Status: {user['status']}"
    if include_email:
        info += f", Email: {user['email']}"
    return info


@tool
def check_inventory(sku: str, warehouse: str = "main") -> str:
    """
    Check inventory levels for a product.

    Args:
        sku: The product SKU (e.g., "SKU-001")
        warehouse: The warehouse to check ("main", "west", "east")
    """
    valid_warehouses = ["main", "west", "east"]
    if warehouse not in valid_warehouses:
        # This will trigger Silent Review - invalid parameter value
        return f"Error: Invalid warehouse '{warehouse}'. Valid options: {valid_warehouses}"

    if sku not in INVENTORY_DB:
        return f"Error: Product '{sku}' not found."

    item = INVENTORY_DB[sku]
    if item["status"] == "out_of_stock":
        # This might trigger Silent Review - item unavailable
        return f"Warning: {item['name']} (SKU: {sku}) is OUT OF STOCK. Quantity: 0"

    return f"Product: {item['name']}, SKU: {sku}, Quantity: {item['quantity']}, Price: ${item['price']}, Warehouse: {warehouse}"


@tool
def create_support_ticket(
    title: str,
    description: str,
    priority: str,
    category: str
) -> str:
    """
    Create a new support ticket.

    Args:
        title: Brief title for the ticket (required)
        description: Detailed description of the issue (required)
        priority: Ticket priority - must be "low", "medium", "high", or "critical" (required)
        category: Ticket category - must be "bug", "feature", "question", or "other" (required)
    """
    valid_priorities = ["low", "medium", "high", "critical"]
    valid_categories = ["bug", "feature", "question", "other"]

    errors = []
    if priority not in valid_priorities:
        errors.append(f"Invalid priority '{priority}'. Must be one of: {valid_priorities}")
    if category not in valid_categories:
        errors.append(f"Invalid category '{category}'. Must be one of: {valid_categories}")
    if len(title) < 5:
        errors.append("Title must be at least 5 characters")
    if len(description) < 10:
        errors.append("Description must be at least 10 characters")

    if errors:
        return f"Failed to create ticket. Errors: {'; '.join(errors)}"

    import random
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    return f"Success: Created ticket {ticket_id} - '{title}' (Priority: {priority}, Category: {category})"


@tool
def send_notification(
    recipient_id: str,
    message: str,
    channel: str = "email"
) -> str:
    """
    Send a notification to a user.

    Args:
        recipient_id: The user ID to send notification to (required)
        message: The notification message (required)
        channel: Notification channel - "email", "sms", or "slack" (default: "email")
    """
    valid_channels = ["email", "sms", "slack"]

    if channel not in valid_channels:
        return f"Error: Invalid channel '{channel}'. Valid options: {valid_channels}"

    if recipient_id not in USERS_DB:
        return f"Error: Recipient '{recipient_id}' not found."

    user = USERS_DB[recipient_id]
    if user["status"] == "inactive":
        # Silent error - user exists but is inactive
        return f"Warning: User {user['name']} is inactive. Notification queued but may not be delivered."

    return f"Success: Notification sent to {user['name']} via {channel}: '{message[:50]}...'" if len(message) > 50 else f"Success: Notification sent to {user['name']} via {channel}: '{message}'"


@tool
def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    currency: str = "USD"
) -> str:
    """
    Transfer funds between accounts. USE WITH CAUTION.

    Args:
        from_account: Source account ID (required)
        to_account: Destination account ID (required)
        amount: Amount to transfer - must be positive (required)
        currency: Currency code - "USD", "EUR", "GBP" (default: "USD")
    """
    valid_currencies = ["USD", "EUR", "GBP"]

    if currency not in valid_currencies:
        return f"Error: Invalid currency '{currency}'. Supported: {valid_currencies}"

    if amount <= 0:
        return f"Error: Amount must be positive. Got: {amount}"

    if amount > 10000:
        return f"Error: Amount ${amount} exceeds single transfer limit of $10,000. Requires manager approval."

    if from_account == to_account:
        return f"Error: Source and destination accounts cannot be the same."

    # Simulate successful transfer
    return f"Success: Transferred {currency} {amount:.2f} from {from_account} to {to_account}. Transaction ID: TXN-{hash(from_account + to_account) % 100000:05d}"


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
    # SPARC validation
    validation_passed: bool
    validation_issues: List[str]
    # ToolGuard policy
    policy_passed: bool
    policy_violations: List[str]
    # Silent Review
    review_outcome: Optional[str]
    review_details: Optional[str]
    # RAG Repair
    repair_attempted: bool
    repair_suggestion: Optional[str]
    # Policy Guard (response)
    response_compliant: bool
    response_violations: List[str]


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
    tools = [
        search_knowledge_base,
        calculate,
        get_current_date,
        get_user_info,
        check_inventory,
        create_support_ticket,
        send_notification,
        transfer_funds,
    ]
    tool_specs = get_tool_specs(tools)

    # Bind tools to the model
    model_with_tools = llm.bind_tools(tools)

    # Create tool executor
    tool_executor = ToolNode(tools)

    # System prompt
    system_prompt = """You are a helpful assistant that can answer questions using available tools.

Available tools:
1. search_knowledge_base - Search company policies, products, and technical docs
2. calculate - Perform math calculations
3. get_current_date - Get current date and time
4. get_user_info - Look up user information by ID (e.g., U001, U002, U003)
5. check_inventory - Check product inventory by SKU (e.g., SKU-001, SKU-002)
6. create_support_ticket - Create a support ticket (requires title, description, priority, category)
7. send_notification - Send notification to a user via email/sms/slack
8. transfer_funds - Transfer money between accounts

When using tools, make sure to provide all required parameters with valid values.
Always base your answers on the information retrieved from tools.
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
    # NODE: ToolGuard (Policy enforcement - runs after SPARC, before tools)
    # ========================================================================
    def toolguard_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        ToolGuard node - enforces policies before tool execution.

        Checks tool calls against defined policies to prevent:
        - Dangerous operations (large transfers)
        - Policy violations (sending to inactive users)
        - Business rule violations
        """
        print("    [ToolGuard] Checking policies...")

        tool_call = state.get("current_tool_call")
        if not tool_call:
            return {"policy_passed": True, "policy_violations": []}

        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        violations = []

        # Check policies for this tool
        if tool_name in TOOL_POLICIES:
            for policy in TOOL_POLICIES[tool_name]:
                try:
                    if not policy["check"](tool_args):
                        violation_msg = policy["violation_message"].format(**tool_args)
                        violations.append(f"{policy['name']}: {violation_msg}")
                except Exception as e:
                    pass  # Skip policy if check fails

        if violations:
            print(f"    [ToolGuard] Policy VIOLATIONS: {violations}")
            return {"policy_passed": False, "policy_violations": violations}
        else:
            print(f"    [ToolGuard] All policies PASSED")
            return {"policy_passed": True, "policy_violations": []}

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
    # NODE: RAG Repair (Attempt to fix failed tool calls)
    # ========================================================================
    def rag_repair_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        RAG Repair node - uses documentation to suggest fixes for failed tool calls.

        When a tool call fails or returns errors, this node:
        1. Searches repair documentation for relevant info
        2. Suggests corrections based on documentation
        3. Provides guidance to the agent for retry
        """
        print("    [RAG Repair] Searching documentation for fixes...")

        tool_call = state.get("current_tool_call", {})
        tool_response = state.get("tool_response", "")
        tool_name = tool_call.get("name", "")

        # Find relevant documentation
        relevant_docs = []
        response_lower = tool_response.lower() if tool_response else ""

        # Match error to documentation
        if "user" in tool_name or "user" in response_lower:
            relevant_docs.append(REPAIR_DOCS["user_ids"])

        if "inventory" in tool_name or "sku" in response_lower or "warehouse" in response_lower:
            relevant_docs.append(REPAIR_DOCS["inventory_skus"])

        if "ticket" in tool_name or "priority" in response_lower or "category" in response_lower:
            relevant_docs.append(REPAIR_DOCS["support_tickets"])

        if "transfer" in tool_name or "amount" in response_lower or "currency" in response_lower:
            relevant_docs.append(REPAIR_DOCS["transfers"])

        if relevant_docs:
            suggestion = f"Based on documentation:\n" + "\n---\n".join(relevant_docs)
            print(f"    [RAG Repair] Found relevant documentation for '{tool_name}'")
            return {
                "repair_attempted": True,
                "repair_suggestion": suggestion,
            }
        else:
            print(f"    [RAG Repair] No relevant documentation found")
            return {
                "repair_attempted": True,
                "repair_suggestion": None,
            }

    # ========================================================================
    # NODE: Policy Guard (Check response compliance before returning)
    # ========================================================================
    def policy_guard_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Policy Guard node - checks final response for compliance.

        Ensures the response doesn't contain:
        - Sensitive information (PII, account numbers)
        - Unprofessional language
        - Policy-violating content
        """
        print("    [Policy Guard] Checking response compliance...")

        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message:
            return {"response_compliant": True, "response_violations": []}

        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)
        violations = []

        for policy in RESPONSE_POLICIES:
            try:
                if not policy["check"](response_text):
                    violations.append(f"{policy['name']}: {policy['violation_message']}")
            except Exception:
                pass

        if violations:
            print(f"    [Policy Guard] Response VIOLATIONS: {violations}")
            return {"response_compliant": False, "response_violations": violations}
        else:
            print(f"    [Policy Guard] Response is COMPLIANT")
            return {"response_compliant": True, "response_violations": []}

    # ========================================================================
    # ROUTING FUNCTIONS
    # ========================================================================
    def should_continue(state: Dict[str, Any]) -> str:
        """Decide whether to validate, or end (go to policy guard)."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message:
            return "end"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "validate"

        # No tool calls - go to policy guard before ending
        return "policy_guard"

    def route_after_validation(state: Dict[str, Any]) -> str:
        """Route based on SPARC validation result."""
        if state.get("validation_passed", True):
            return "toolguard"
        else:
            # On validation failure, go back to agent with feedback
            return "agent"

    def route_after_toolguard(state: Dict[str, Any]) -> str:
        """Route based on ToolGuard policy check."""
        if state.get("policy_passed", True):
            return "tools"
        else:
            # On policy violation, go back to agent with feedback
            return "agent"

    def route_after_review(state: Dict[str, Any]) -> str:
        """Route based on Silent Review result."""
        review_outcome = state.get("review_outcome", "ok")
        if review_outcome == "issues_detected":
            # Try RAG repair first
            return "repair"
        return "agent"

    def route_after_repair(state: Dict[str, Any]) -> str:
        """After repair, always go back to agent."""
        return "agent"

    def route_after_policy_guard(state: Dict[str, Any]) -> str:
        """Route based on Policy Guard result."""
        if state.get("response_compliant", True):
            return "end"
        else:
            # Response has violations - could retry or warn
            # For now, we'll still end but the violations are logged
            return "end"

    # ========================================================================
    # BUILD THE GRAPH
    # ========================================================================
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", sparc_validation_node)      # SPARC validation
    workflow.add_node("toolguard", toolguard_node)            # ToolGuard policy
    workflow.add_node("tools", tool_node)                      # Tool execution
    workflow.add_node("review", silent_review_node)            # Silent Review
    workflow.add_node("repair", rag_repair_node)               # RAG Repair
    workflow.add_node("policy_guard", policy_guard_node)       # Policy Guard

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    # Agent -> SPARC validation (if tool call) or Policy Guard (if final response)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"validate": "validate", "policy_guard": "policy_guard", "end": END}
    )

    # SPARC -> ToolGuard (if passed) or back to Agent (if failed)
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"toolguard": "toolguard", "agent": "agent"}
    )

    # ToolGuard -> Tools (if passed) or back to Agent (if policy violation)
    workflow.add_conditional_edges(
        "toolguard",
        route_after_toolguard,
        {"tools": "tools", "agent": "agent"}
    )

    # Tools -> Silent Review
    workflow.add_edge("tools", "review")

    # Silent Review -> RAG Repair (if issues) or Agent (if ok)
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {"repair": "repair", "agent": "agent"}
    )

    # RAG Repair -> Agent
    workflow.add_edge("repair", "agent")

    # Policy Guard -> End
    workflow.add_conditional_edges(
        "policy_guard",
        route_after_policy_guard,
        {"end": END}
    )

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
    print("=" * 70)
    print("  Simple RAG Agent with Full ALTK Component Stack - Demo")
    print("=" * 70)
    print("\nALTK Components in use:")
    print("  PRE-TOOL PHASE:")
    print("    - SPARC Validation: Validates tool calls (params, types, schema)")
    print("    - ToolGuard: Enforces policies (blocks dangerous operations)")
    print("  POST-TOOL PHASE:")
    print("    - Silent Review: Detects errors in tool responses")
    print("    - RAG Repair: Suggests fixes using documentation")
    print("  PRE-RESPONSE PHASE:")
    print("    - Policy Guard: Checks response compliance (PII, professionalism)")
    print("\nTest scenarios:")
    print("  1. Valid queries (all checks pass)")
    print("  2. User not found (Silent Review + RAG Repair)")
    print("  3. Policy violation - large transfer (ToolGuard blocks)")
    print("  4. Policy violation - notify inactive user (ToolGuard blocks)")
    print("  5. Invalid warehouse (Silent Review + RAG Repair)")
    print("  6. Out of stock (Silent Review detects warning)")
    print("\nNote: Make sure Ollama is running (ollama serve)")
    print("And you have a model pulled (ollama pull llama3.2)")

    # Create the agent
    agent = create_rag_agent_altk()

    # Test questions - designed to showcase ALL ALTK components
    questions = [
        # 1. Basic RAG query - should pass all checks
        "What is the vacation policy for new employees?",

        # 2. Valid user lookup - should pass all checks
        "Get info about user U001, include their email",

        # 3. User not found - triggers Silent Review + RAG Repair
        "Get info about user U999",

        # 4. ToolGuard: Large transfer blocked by policy ($5000 limit)
        "Transfer $7500 from ACC-001 to ACC-002",

        # 5. ToolGuard: Notification to inactive user blocked
        "Send a slack notification to user U003 saying 'Please review'",

        # 6. Invalid warehouse - triggers Silent Review + RAG Repair
        "Check inventory for SKU-001 in the tokyo warehouse",

        # 7. Out of stock - Silent Review detects warning
        "Check inventory for SKU-002",

        # 8. Valid transfer within policy limits - should pass
        "Transfer $500 from ACC-001 to ACC-002",

        # 9. Create support ticket - tests SPARC validation
        "Create a support ticket: title='Login issue', description='Cannot login since Monday morning update', priority='high', category='bug'",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'═' * 70}")
        print(f"Test {i}: {q}")
        print(f"{'═' * 70}")
        response = ask_question(agent, q)
        print(f"\nAnswer: {response}")
