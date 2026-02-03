"""
Try ALTK Prebuilt Agents

This example shows how easy it is to use ALTK with LangGraph
using the prebuilt agents.

Run:
    cd /Users/anu/agent-lifecycle-toolkit

    # Set your OpenAI API key
    export OPENAI_API_KEY=your-key-here

    # Run
    python examples/try_prebuilt_agents.py

Requirements:
    pip install langgraph langchain-core langchain-openai
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check dependencies
try:
    from langchain_core.tools import tool
except ImportError:
    print("Please install: pip install langchain-core")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph
except ImportError:
    print("Please install: pip install langgraph")
    sys.exit(1)


# =============================================================================
# Define Tools
# =============================================================================

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "San Francisco", "London")
        units: Temperature units - "celsius" or "fahrenheit"
    """
    # Simulated weather data
    weather_data = {
        "san francisco": {"temp": 18, "conditions": "foggy"},
        "new york": {"temp": 22, "conditions": "sunny"},
        "london": {"temp": 15, "conditions": "rainy"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp"]
        if units == "fahrenheit":
            temp = int(temp * 9/5 + 32)
        return f"Weather in {city}: {temp}°{'F' if units == 'fahrenheit' else 'C'}, {data['conditions']}"

    return f"Weather data not available for {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
    """
    import math

    # Safe eval with math functions
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def search_products(query: str, max_results: int = 5) -> str:
    """Search for products in the catalog.

    Args:
        query: Search query
        max_results: Maximum number of results (1-10)
    """
    # Simulated product data
    products = [
        {"name": "Laptop Pro", "price": 1299, "category": "electronics"},
        {"name": "Wireless Mouse", "price": 49, "category": "electronics"},
        {"name": "Mechanical Keyboard", "price": 149, "category": "electronics"},
        {"name": "Monitor 27\"", "price": 399, "category": "electronics"},
        {"name": "USB Hub", "price": 29, "category": "accessories"},
    ]

    # Simple search
    query_lower = query.lower()
    matches = [p for p in products if query_lower in p["name"].lower() or query_lower in p["category"]]
    matches = matches[:max_results]

    if not matches:
        return f"No products found for '{query}'"

    result = f"Found {len(matches)} products:\n"
    for p in matches:
        result += f"  - {p['name']}: ${p['price']}\n"
    return result


# =============================================================================
# Example 1: Using a Mock Model (No API Key Needed)
# =============================================================================

def example_with_mock_model():
    """Demo using a mock model - no API key required."""
    print("\n" + "=" * 60)
    print("Example 1: ALTK Agent with Mock Model")
    print("=" * 60)

    from altk.integrations.langgraph import create_altk_react_agent

    # Create a simple mock model for testing
    class MockChatModel:
        """Mock model that always calls get_weather."""

        def __init__(self):
            self.tools = []
            self.call_count = 0

        def bind_tools(self, tools):
            self.tools = tools
            return self

        def invoke(self, messages):
            from langchain_core.messages import AIMessage

            self.call_count += 1

            # First call: call the tool
            if self.call_count == 1:
                return AIMessage(
                    content="Let me check the weather for you.",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "San Francisco", "units": "celsius"},
                    }]
                )

            # Second call: respond with result
            return AIMessage(
                content="The weather in San Francisco is 18°C and foggy. Perfect for a walk!"
            )

    # Create agent with mock model
    mock_model = MockChatModel()

    agent = create_altk_react_agent(
        model=mock_model,
        tools=[get_weather, calculate, search_products],
        validation_track="syntax",  # Fast, no LLM needed
        review_enabled=False,       # Skip review for simplicity
    )

    print("\nInvoking agent with: 'What's the weather in San Francisco?'")
    print("-" * 60)

    result = agent.invoke({
        "messages": [("user", "What's the weather in San Francisco?")]
    })

    print("\nMessages in conversation:")
    for i, msg in enumerate(result.get("messages", [])):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        print(f"  {i+1}. [{role}] {content[:100]}...")

    print("\nValidation result:", result.get("validation_result"))

    return result


# =============================================================================
# Example 2: Using Real OpenAI Model
# =============================================================================

def example_with_openai():
    """Demo using real OpenAI model."""
    print("\n" + "=" * 60)
    print("Example 2: ALTK Agent with OpenAI")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nSkipping: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=your-key-here")
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("\nSkipping: langchain-openai not installed")
        print("Install with: pip install langchain-openai")
        return None

    from altk.integrations.langgraph import create_altk_react_agent

    # Create agent with real model
    agent = create_altk_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[get_weather, calculate, search_products],
        validation_track="syntax",
        review_enabled=True,
        messages_modifier="You are a helpful assistant. Be concise.",
    )

    print("\nInvoking agent with: 'What's 25 * 4 and what's the weather in NYC?'")
    print("-" * 60)

    result = agent.invoke({
        "messages": [("user", "What's 25 * 4 and what's the weather in NYC?")]
    })

    print("\nFinal response:")
    final_msg = result["messages"][-1]
    print(f"  {final_msg.content}")

    print("\nValidation results captured:", result.get("validation_result") is not None)
    print("Review results captured:", result.get("review_result") is not None)

    return result


# =============================================================================
# Example 3: Different Agent Types
# =============================================================================

def example_agent_types():
    """Show different prebuilt agent types."""
    print("\n" + "=" * 60)
    print("Example 3: Different Agent Types")
    print("=" * 60)

    from altk.integrations.langgraph import (
        create_altk_react_agent,
        create_altk_safe_agent,
        create_altk_fast_agent,
        create_altk_tool_agent,
    )

    # Mock model for demo
    class MockModel:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content="Done!")

    model = MockModel()
    tools = [get_weather]

    print("\nAvailable prebuilt agents:")
    print("-" * 60)

    print("""
    1. create_altk_react_agent
       - Full-featured ReAct agent
       - Configurable validation, review, repair
       - Use for: General purpose applications

    2. create_altk_safe_agent
       - Maximum validation (slow_track)
       - Policy enforcement + RAG repair
       - Use for: High-stakes, enterprise apps

    3. create_altk_fast_agent
       - Syntax-only validation
       - No review or repair
       - Use for: Speed-critical applications

    4. create_altk_tool_agent
       - Simple tool calling
       - Basic validation + review
       - Use for: Simple tool-calling tasks
    """)

    print("Creating each agent type...")

    # Show that they all work
    agents = {
        "react": create_altk_react_agent(model, tools, validation_track="syntax", review_enabled=False),
        "fast": create_altk_fast_agent(model, tools),
        "tool": create_altk_tool_agent(model, tools, review_enabled=False),
    }

    for name, agent in agents.items():
        print(f"  ✓ {name} agent created successfully")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALTK Prebuilt Agents Demo")
    print("=" * 60)

    # Example 1: Works without any API key
    example_with_mock_model()

    # Example 2: Requires OpenAI API key
    example_with_openai()

    # Example 3: Show agent types
    example_agent_types()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("""
Next steps:
  1. Set OPENAI_API_KEY to try with real models
  2. Try different validation tracks (syntax, fast_track, slow_track)
  3. Enable review and repair for production use
  4. See README.md for full documentation
    """)
