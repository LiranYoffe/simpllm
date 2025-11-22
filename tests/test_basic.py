"""Basic smoke tests for simpllm package."""

from pydantic import Field

from simpllm import (
    AnthropicWrapper,
    AssistantMessage,
    AssistantTextBlock,
    BaseTool,
    GeminiWrapper,
    ThinkingBlock,
    Usage,
    UserMessage,
    UserTextBlock,
    pydantic_to_function_declaration,
)


def test_user_message_from_text():
    """Test creating user message from text."""
    msg = UserMessage.from_text("Hello!")
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], UserTextBlock)
    assert msg.content[0].text == "Hello!"


def test_assistant_message_to_aggregated():
    """Test converting assistant message to aggregated response."""
    msg = AssistantMessage(
        content=[
            AssistantTextBlock(text="Hello "),
            ThinkingBlock(text="I should be polite", signature=None),
            AssistantTextBlock(text="there!"),
        ],
        usage=Usage(
            uncached_input_tokens=100,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=0,
            output_tokens=20,
        ),
    )

    aggregated = msg.to_aggregated_response()
    assert aggregated.text == "Hello there!"
    assert aggregated.thoughts == "I should be polite"
    assert len(aggregated.tool_calls) == 0
    assert aggregated.usage.output_tokens == 20


def test_tool_declaration_conversion():
    """Test converting Pydantic model to tool declaration."""

    class TestTool(BaseTool):
        """A test tool."""

        param1: str = Field(description="First parameter")
        param2: int = Field(description="Second parameter")

    # Gemini format
    decl = pydantic_to_function_declaration(TestTool, schema_key="parameters")
    assert decl["name"] == "TestTool"
    assert decl["description"] == "A test tool."
    assert "parameters" in decl

    # Anthropic format
    decl_anthropic = pydantic_to_function_declaration(TestTool, schema_key="input_schema")
    assert decl_anthropic["name"] == "TestTool"  # Both use __tool_name__
    assert "input_schema" in decl_anthropic


def test_wrapper_initialization():
    """Test that wrappers can be initialized."""
    gemini = GeminiWrapper(model="gemini-2.5-flash", system_prompt="Test")
    assert gemini.provider == "google"
    assert gemini.model == "gemini-2.5-flash"

    anthropic = AnthropicWrapper(model="claude-sonnet-4-5-20250929", system_prompt="Test")
    assert anthropic.provider == "anthropic"
    assert anthropic.thinking_budget == 4096


def test_wrapper_with_tools():
    """Test wrapper initialization with tools."""

    class MockTool(BaseTool, tool_name="mock_tool"):
        """A mock tool."""

        arg: str = Field(description="Test argument")

    wrapper = GeminiWrapper(model="gemini-2.5-flash", system_prompt="Test", tools=[MockTool])

    assert wrapper.tools_map is not None
    assert "mock_tool" in wrapper.tools_map
    assert wrapper.tools_map["mock_tool"] == MockTool
