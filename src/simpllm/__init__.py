"""
simpllm - Simple LLM wrapper library for Anthropic Claude and Google Gemini.

Provides a unified interface for interacting with multiple LLM providers
with support for streaming, thinking/reasoning, tool calling, and prompt caching.
"""

__version__ = "0.1.1"

# Message types and blocks
from simpllm.messages import (
    UserTextBlock,
    AssistantTextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserContentBlockType,
    AssistantContentBlockType,
    UserMessage,
    AssistantMessage,
    Usage,
    MessageType,
    MessageRootModel,
    AggregatedResponse,
)

# Provider wrappers
from simpllm.wrappers import (
    ProviderWrapper,
    GeminiWrapper,
    AnthropicWrapper,
)

# Type unions
from simpllm.types import ProviderWrapperType

# Utilities
from simpllm.utils import (
    BaseTool,
    get_logger,
    pydantic_to_function_declaration,
    ToolDeclaration,
    ToolDeclarationAnthropic,
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
    "get_logger",
    "pydantic_to_function_declaration",
    "ToolDeclaration",
    "ToolDeclarationAnthropic",
]
