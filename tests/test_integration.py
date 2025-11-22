"""Integration tests that make actual API calls."""

import os
import pytest
from pydantic import Field

from simpllm import (
    UserMessage,
    GeminiWrapper,
    BaseTool,
)


# Skip all tests in this file if GOOGLE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set in environment"
)


@pytest.mark.asyncio
async def test_gemini_basic_text():
    """Test basic text generation with Gemini."""
    wrapper = GeminiWrapper(
        model="gemini-2.5-flash",
        system_prompt="You are a helpful assistant. Be concise.",
    )

    messages = [UserMessage.from_text("What is 2+2? Answer with just the number.")]
    response = await wrapper.invoke(messages)

    aggregated = response.to_aggregated_response()
    assert aggregated.text is not None
    assert len(aggregated.text) > 0
    assert "4" in aggregated.text

    # Check usage tracking
    assert aggregated.usage is not None
    assert aggregated.usage.output_tokens > 0
    print(f"\nUsage: {aggregated.usage}")


# @pytest.mark.asyncio
# async def test_gemini_with_thinking():
#     """Test Gemini with thinking enabled (requires paid tier)."""
#     wrapper = GeminiWrapper(
#         model="gemini-2.5-flash",
#         system_prompt="You are a math tutor.",
#         thinking_budget=1000,
#     )
#
#     messages = [UserMessage.from_text("What is 15 * 23?")]
#     response = await wrapper.invoke(messages)
#
#     aggregated = response.to_aggregated_response()
#     assert aggregated.text is not None
#     assert "345" in aggregated.text
#
#     # May or may not have thoughts depending on model behavior
#     print(f"\nThoughts: {aggregated.thoughts}")
#     print(f"Response: {aggregated.text}")


@pytest.mark.asyncio
async def test_gemini_tool_calling():
    """Test tool calling with Gemini."""

    class CalculatorTool(BaseTool):
        """Perform basic arithmetic operations."""

        operation: str = Field(description="Operation: add, subtract, multiply, divide")
        a: float = Field(description="First number")
        b: float = Field(description="Second number")

        async def invoke(self, context) -> str:
            ops = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y,
            }
            result = ops[self.operation](self.a, self.b)
            return f"Result: {result}"

    wrapper = GeminiWrapper(
        model="gemini-2.5-flash",
        system_prompt="You are a calculator assistant. Use the calculator tool for arithmetic.",
        tools=[CalculatorTool],
    )

    messages = [UserMessage.from_text("What is 17 times 19?")]
    response = await wrapper.invoke(messages)

    aggregated = response.to_aggregated_response()

    # Should have requested a tool call
    assert len(aggregated.tool_calls) > 0
    tool_call = aggregated.tool_calls[0]

    assert tool_call.name == "CalculatorTool"
    assert tool_call.args["operation"] == "multiply"
    assert tool_call.args["a"] == 17
    assert tool_call.args["b"] == 19

    print(f"\nTool call: {tool_call.name}")
    print(f"Args: {tool_call.args}")


@pytest.mark.asyncio
async def test_gemini_token_counting():
    """Test token counting functionality."""
    wrapper = GeminiWrapper(
        model="gemini-2.5-flash",
        system_prompt="You are a helpful assistant.",
    )

    messages = [UserMessage.from_text("Hello! How are you today?")]
    token_count = await wrapper.count_input_tokens(messages)

    assert token_count > 0
    assert token_count < 100  # Should be a small message
    print(f"\nToken count: {token_count}")


@pytest.mark.asyncio
async def test_gemini_conversation():
    """Test multi-turn conversation."""
    wrapper = GeminiWrapper(
        model="gemini-2.5-flash",
        system_prompt="You are a helpful assistant.",
    )

    # First turn
    messages = [UserMessage.from_text("My name is Alice.")]
    response1 = await wrapper.invoke(messages)

    # Second turn - test conversation memory
    messages.append(response1)
    messages.append(UserMessage.from_text("What is my name?"))
    response2 = await wrapper.invoke(messages)

    aggregated = response2.to_aggregated_response()
    assert "Alice" in aggregated.text or "alice" in aggregated.text.lower()
    print(f"\nConversation response: {aggregated.text}")