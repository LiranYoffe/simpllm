"""
simpllm - Simple LLM wrapper library for Anthropic Claude and Google Gemini.

Provides a unified interface for interacting with multiple LLM providers
with support for streaming, thinking/reasoning, tool calling, and prompt caching.
"""

__version__ = "0.1.2"

# Message types and blocks
from simpllm.messages import (
    AggregatedResponse,
    AssistantContentBlockType,
    AssistantMessage,
    AssistantTextBlock,
    MessageRootModel,
    MessageType,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
    UserContentBlockType,
    UserMessage,
    UserTextBlock,
)
from simpllm.tools import BaseTool

# Type unions
from simpllm.types import ProviderWrapperType

# Utilities
from simpllm.utils import (
    ToolDeclaration,
    ToolDeclarationAnthropic,
    pydantic_to_function_declaration,
)

# Provider wrappers
from simpllm.wrappers import (
    AnthropicWrapper,
    GeminiWrapper,
    ProviderWrapper,
)

__all__ = [
    # Version
    "__version__",
    # Messages
    "UserTextBlock",
    "AssistantTextBlock",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "UserContentBlockType",
    "AssistantContentBlockType",
    "UserMessage",
    "AssistantMessage",
    "Usage",
    "MessageType",
    "MessageRootModel",
    "AggregatedResponse",
    # Wrappers
    "ProviderWrapper",
    "GeminiWrapper",
    "AnthropicWrapper",
    "ProviderWrapperType",
    # Utilities
    "BaseTool",
    "pydantic_to_function_declaration",
    "ToolDeclaration",
    "ToolDeclarationAnthropic",
]
