"""Message types and content blocks for LLM interactions."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, RootModel


class UserTextBlock(BaseModel):
    """User text content block."""

    type: Literal["text"] = "text"
    text: str


class AssistantTextBlock(BaseModel):
    """Assistant text response block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Extended thinking/reasoning block with signature."""

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None


class ToolCallBlock(BaseModel):
    """Tool/function call block."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str | None
    name: str
    args: dict[str, Any]
    index: int


class ToolResultBlock(BaseModel):
    """Tool execution result block."""

    type: Literal["tool_result"] = "tool_result"
    result: str
    is_error: bool = False
    tool_call: ToolCallBlock


UserContentBlockType = Annotated[UserTextBlock | ToolResultBlock, Field(discriminator="type")]

AssistantContentBlockType = Annotated[AssistantTextBlock | ThinkingBlock | ToolCallBlock, Field(discriminator="type")]


class UserMessage(BaseModel):
    """User message with content blocks."""

    role: Literal["user"] = "user"
    content: list[UserContentBlockType]

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Create user message from plain text."""
        return cls(content=[UserTextBlock(text=text)])

    @classmethod
    def from_tool_results(cls, tool_results: Sequence[ToolResultBlock]) -> Self:
        """Create user message from tool results."""
        tool_results_sorted = sorted(tool_results, key=lambda tr: tr.tool_call.index)
        return cls(content=list(tool_results_sorted))


class Usage(BaseModel):
    """Token usage tracking."""

    uncached_input_tokens: int  # Regular input tokens that are NOT read from cache NOR explicitly written to cache
    cache_read_input_tokens: int  # Input tokens read from cache
    cache_creation_input_tokens: int  # Relevant ONLY for Anthropic
    output_tokens: int


class AssistantMessage(BaseModel):
    """Assistant message with content blocks and usage stats."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContentBlockType]
    usage: Usage | None = None
    structured_output: BaseModel | None = None

    def to_aggregated_response(self) -> "AggregatedResponse":
        """Convert to aggregated response format."""
        text_lst: list[str] = []
        thoughts_lst: list[str] = []
        tool_calls: list[ToolCallBlock] = []
        for block in self.content:
            if isinstance(block, AssistantTextBlock):
                text_lst.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thoughts_lst.append(block.text)
            elif isinstance(block, ToolCallBlock):
                tool_calls.append(block)

        return AggregatedResponse(
            text="".join(text_lst),
            thoughts="".join(thoughts_lst),
            tool_calls=tool_calls,
            usage=self.usage,
            structured_output=self.structured_output,
        )


MessageType = Annotated[UserMessage | AssistantMessage, Field(discriminator="role")]
MessageRootModel = RootModel[MessageType]


class AggregatedResponse(BaseModel):
    """Aggregated response with text, thoughts, tool calls, and usage."""

    text: str
    thoughts: str
    tool_calls: list[ToolCallBlock]
    structured_output: BaseModel | None = None
    usage: Usage | None = None
