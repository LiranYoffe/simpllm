# simpllm

A simple, unified Python library for interacting with multiple LLM providers (Anthropic Claude and Google Gemini) with support for streaming, extended thinking, tool calling, and prompt caching.

## Features

- **Unified Interface**: Single API for both Anthropic Claude and Google Gemini
- **Extended Thinking**: Support for reasoning/thinking blocks in responses
- **Tool Calling**: Function calling with automatic schema generation from Pydantic models
- **Prompt Caching**: Automatic prompt caching support (Anthropic)
- **Token Usage Tracking**: Detailed token usage statistics including cache hits
- **Async First**: Built on asyncio for efficient concurrent operations
- **Type Safe**: Full type hints and Pydantic models throughout

## Installation

```bash
uv add simpllm
```

## Quick Start

### Using Anthropic Claude

```python
import asyncio
from simpllm import AnthropicWrapper, UserMessage

async def main():
    wrapper = AnthropicWrapper(
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a helpful assistant.",
        thinking_budget=4096,
    )

    messages = [UserMessage.from_text("What is 2+2?")]
    response = await wrapper.invoke(messages)

    aggregated = response.to_aggregated_response()
    print(f"Response: {aggregated.text}")
    print(f"Thoughts: {aggregated.thoughts}")
    print(f"Usage: {aggregated.usage}")

asyncio.run(main())
```

### Using Google Gemini

```python
import asyncio
from simpllm import GeminiWrapper, UserMessage

async def main():
    wrapper = GeminiWrapper(
        model="gemini-2.5-flash",
        system_prompt="You are a helpful assistant.",
        thinking_budget=1000,
    )

    messages = [UserMessage.from_text("Explain quantum computing")]
    response = await wrapper.invoke(messages)

    aggregated = response.to_aggregated_response()
    print(f"Response: {aggregated.text}")

asyncio.run(main())
```

## Tool Calling

Define tools by inheriting from `BaseTool`:

```python
from pydantic import Field
from simpllm import BaseTool, AnthropicWrapper, UserMessage

class CalculatorTool(BaseTool):
    """Perform basic arithmetic operations."""

    operation: str = Field(description="Operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

    async def invoke(self, context) -> str:
        """Execute the tool."""
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y,
        }
        result = ops[self.operation](self.a, self.b)
        return f"Result: {result}"

async def main():
    wrapper = AnthropicWrapper(
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a calculator assistant.",
        tools=[CalculatorTool],
    )

    messages = [UserMessage.from_text("What is 15 times 23?")]
    response = await wrapper.invoke(messages)

    # Check for tool calls
    aggregated = response.to_aggregated_response()
    if aggregated.tool_calls:
        for tool_call in aggregated.tool_calls:
            print(f"Tool: {tool_call.name}")
            print(f"Args: {tool_call.args}")
```

**Note**: The `__tool_name__` class attribute is automatically set to the class name (`"CalculatorTool"`) by `BaseTool.__init_subclass__`. You can customize it:

```python
class CalculatorTool(BaseTool, tool_name="calculator"):
    """Custom tool name example."""
    # ... fields ...
```

## Message Management

### Creating Messages

```python
from simpllm import UserMessage, UserTextBlock, ToolResultBlock

# From plain text
msg = UserMessage.from_text("Hello!")

# From tool results
tool_results = [ToolResultBlock(...)]
msg = UserMessage.from_tool_results(tool_results)

# Manual construction
msg = UserMessage(content=[UserTextBlock(text="Custom message")])
```

### Conversation History

```python
messages = [
    UserMessage.from_text("Hello!"),
    # ... add assistant responses to conversation
    UserMessage.from_text("Tell me more"),
]

response = await wrapper.invoke(messages)
```

## Configuration

### Anthropic Claude

```python
wrapper = AnthropicWrapper(
    model="claude-sonnet-4-5-20250929",
    system_prompt="Your system prompt",
    thinking_budget=4096,           # Thinking tokens budget
    max_output_tokens=64_000,       # Max output tokens
    use_cache=True,                 # Enable prompt caching
    interleaved_thinking=True,      # Enable interleaved thinking
    base_uri=None,                  # Optional custom API base URL
    api_key_env_var=None,           # Custom API key env var name
)
```

### Google Gemini

```python
wrapper = GeminiWrapper(
    model="gemini-2.5-flash",
    system_prompt="Your system prompt",
    thinking_budget=1000,           # Thinking tokens budget (-1 is default)
)
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Anthropic API key (or use `api_key_env_var` parameter)
- `GOOGLE_API_KEY`: Google API key for Gemini

## Token Usage

All responses include detailed token usage:

```python
response = await wrapper.invoke(messages)
usage = response.usage

print(f"Uncached input: {usage.uncached_input_tokens}")
print(f"Cache read: {usage.cache_read_input_tokens}")
print(f"Cache creation: {usage.cache_creation_input_tokens}")
print(f"Output: {usage.output_tokens}")
```

## Advanced Features

### Structured Output (Gemini)

```python
from pydantic import BaseModel

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

wrapper = GeminiWrapper(
    model="gemini-2.5-flash",
    system_prompt="Analyze sentiment",
    output_schema=SentimentResponse,
)

response = await wrapper.invoke(messages)
structured_output = response.structured_output  # SentimentResponse instance
```

### Token Counting

```python
messages = [UserMessage.from_text("Hello!")]
token_count = await wrapper.count_input_tokens(messages)
print(f"This will use approximately {token_count} input tokens")
```

## Architecture

- **Messages**: Unified message format with content blocks (text, thinking, tool calls, tool results)
- **Wrappers**: Provider-specific implementations with common interface
- **Stateless**: Wrappers don't maintain conversation state - pass full message history
- **Type Safe**: Pydantic models throughout for validation and serialization

## Requirements

- Python 3.12+
- anthropic >= 0.60.0
- google-genai >= 1.26.0
- pydantic >= 2.0.0
- tenacity >= 8.0.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

