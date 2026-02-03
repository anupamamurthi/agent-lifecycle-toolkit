"""
Simple LangGraph Agent with Tools and RAG

A straightforward example of a LangGraph-based agent that:
- Uses tools to perform actions
- Retrieves context from documents (RAG)
- Answers user questions
"""

from typing import Annotated, TypedDict, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

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

    # Simple keyword matching for demonstration
    results = []

    if any(word in query_lower for word in ["policy", "work", "vacation", "remote", "sick", "equipment"]):
        results.append(KNOWLEDGE_BASE["company_policy"])

    if any(word in query_lower for word in ["product", "price", "feature", "support", "trial", "plan"]):
        results.append(KNOWLEDGE_BASE["product_info"])

    if any(word in query_lower for word in ["api", "technical", "rate", "format", "auth", "sdk", "webhook"]):
        results.append(KNOWLEDGE_BASE["technical_docs"])

    if not results:
        # Return all documents if no specific match
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
        # Safe evaluation of mathematical expressions
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
# AGENT SETUP
# ============================================================================

def create_rag_agent():
    """Create a LangGraph agent with tools and RAG capabilities."""

    # Initialize the LLM (Ollama runs locally)
    llm = ChatOllama(
        model="llama3.2",  # or "mistral", "codellama", etc.
        temperature=0,
    )

    # Define the tools available to the agent
    tools = [
        search_knowledge_base,
        calculate,
        get_current_date,
    ]

    # System prompt that guides the agent's behavior
    system_prompt = """You are a helpful assistant that can answer questions using available tools.

When answering questions:
1. If the question is about company policies, products, or technical information,
   use the search_knowledge_base tool to find relevant information.
2. If the question involves calculations, use the calculate tool.
3. If the question involves dates or times, use the get_current_date tool.

Always base your answers on the information retrieved from tools when available.
Be concise and helpful in your responses."""

    # Create the ReAct agent using LangGraph
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def ask_question(agent, question: str) -> str:
    """
    Ask the agent a question and get a response.

    Args:
        agent: The LangGraph agent
        question: The question to ask

    Returns:
        The agent's response
    """
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Get the final response from the agent
    final_message = result["messages"][-1]
    return final_message.content


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Simple RAG Agent with Tools - Demo")
    print("=" * 60)
    print("\nNote: Make sure Ollama is running (ollama serve)")
    print("And you have a model pulled (ollama pull llama3.2)\n")

    # Create the agent
    agent = create_rag_agent()

    # Questions that will use different tools and documents
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
        print(f"A: {response}")
