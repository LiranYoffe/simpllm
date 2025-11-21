"""LLM provider wrapper implementations."""

import asyncio
import logging
from abc import ABC, abstractmethod
from os import environ
from typing import Any, Literal

import anthropic
import anthropic.types as anthropic_types
import anthropic.types.beta as anthropic_types_beta
from anthropic.types import TextBlockParam
from anthropic.types.beta import BetaMessageParam, BetaToolParam
from google import genai
from google.genai import types as google_types
from pydantic import BaseModel, PrivateAttr
from tenacity import retry, wait_fixed, before_sleep_log, stop_after_attempt

from simpllm.messages import (
    UserMessage,
    AssistantMessage,
    Usage,
    UserTextBlock,
    ToolResultBlock,
    AssistantTextBlock,
    ThinkingBlock,
    ToolCallBlock,
    MessageType,
)
from simpllm.utils import pydantic_to_function_declaration, BaseTool

logger = logging.getLogger("simpllm")


class ProviderWrapper[Client, Message, Response, ToolDeclaration](BaseModel, ABC):
    """
    Abstract base class for LLM provider wrappers.

    Provides common functionality for interacting with different LLM providers
    (Anthropic Claude, Google Gemini, etc.) with a unified interface.

    Type Parameters:
        Client: Provider's client type
        Message: Provider's message type
        Response: Provider's response type
        ToolDeclaration: Provider's tool declaration type

    Note:
        Tools should be BaseTool subclasses with a __tool_name__ class attribute
        and an invoke() method. The wrapper uses these models for function calling.
    """

    provider: str
    model: str
    system_prompt: str
    tools: list[type[BaseTool]] | None = None
    output_schema: type[BaseModel] | None = None
    use_cache: bool = True  # Relevant to Anthropic only

    _client: Client = PrivateAttr()
    _tools_declarations: list[ToolDeclaration] | None = PrivateAttr(default=None)
    _tools_map: dict[str, type[BaseTool]] | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        """Initialize client and tool declarations after model creation."""
        if self.tools:
            self._tools_map = {getattr(tool, "__tool_name__", tool.__name__): tool for tool in self.tools}
            self._tools_declarations = list(map(self.to_native_tool_declaration, self.tools))

        self._client = self.setup_client()

    @abstractmethod
    def setup_client(self) -> Client:
        """Set up and return provider client."""
        ...

    @staticmethod
    @abstractmethod
    def to_native_tool_declaration(tool: type[BaseTool]) -> ToolDeclaration:
        """Convert BaseTool to provider's tool declaration format."""
        ...

    @abstractmethod
    def native_response_to_assistant_message(self, response: Response) -> AssistantMessage:
        """Convert provider's response to AssistantMessage."""
        ...

    # noinspection PyMethodMayBeStatic
    def native_response_to_structured_output(self, response: Response) -> BaseModel | None:
        """Extract structured output from provider's response (if supported)."""
        return None

    @abstractmethod
    def native_response_to_usage(self, response: Response) -> Usage:
        """Extract token usage from provider's response."""
        ...

    @abstractmethod
    def to_native_messages(self, messages: list[MessageType]) -> list[Message]:
        """Convert messages to provider-native format."""
        ...

    @abstractmethod
    async def generate_native_response(self, messages: list[MessageType]) -> Response:
        """Generate response from provider (stateless)."""
        ...

    @abstractmethod
    async def count_input_tokens(self, messages: list[MessageType]) -> int:
        """Count input tokens for given messages."""
        ...

    async def invoke(self, messages: list[MessageType]) -> AssistantMessage:
        """
        Invoke LLM with given messages (stateless).

        Args:
            messages: Conversation messages to send to LLM

        Returns:
            Assistant's response message

        Note:
            Caller is responsible for appending response to state
        """
        response = await self.generate_native_response(messages)
        assistant_msg = self.native_response_to_assistant_message(response)
        return assistant_msg

    @property
    def tools_map(self) -> dict[str, type[BaseTool]] | None:
        """Get mapping of tool names to tool classes."""
        return self._tools_map


class GeminiWrapper(
    ProviderWrapper[
        genai.Client,
        google_types.Content,
        google_types.GenerateContentResponse,
        google_types.FunctionDeclaration,
    ]
):
    """Google Gemini LLM wrapper."""

    provider: Literal["google"] = "google"
    thinking_budget: int = -1
    _config: google_types.GenerateContentConfig = PrivateAttr()

    def setup_client(self) -> genai.Client:
        """Set up Gemini client with configuration."""
        self._config = google_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=2**16,
            tools=(
                [google_types.Tool(function_declarations=self._tools_declarations)]
                if self._tools_declarations
                else None
            ),
            thinking_config=google_types.ThinkingConfig(
                include_thoughts=True, thinking_budget=self.thinking_budget
            ),
            response_mime_type="application/json" if self.output_schema else None,
            response_schema=self.output_schema,
        )
        return genai.Client()

    @staticmethod
    def to_native_tool_declaration(tool: type[BaseTool]) -> google_types.FunctionDeclaration:
        """Convert BaseTool to Gemini function declaration."""
        return google_types.FunctionDeclaration.model_validate(pydantic_to_function_declaration(tool))

    # noinspection PyMethodMayBeStatic
    def native_response_to_usage(self, response: google_types.GenerateContentResponse) -> Usage:
        """Extract token usage from Gemini response."""
        usage_metadata = response.usage_metadata
        cached_content_token_count = usage_metadata.cached_content_token_count or 0
        prompt_token_count = usage_metadata.prompt_token_count or 0
        thoughts_token_count = usage_metadata.thoughts_token_count or 0
        candidates_token_count = usage_metadata.candidates_token_count or 0

        usage = Usage(
            uncached_input_tokens=prompt_token_count - cached_content_token_count,
            cache_read_input_tokens=cached_content_token_count,
            cache_creation_input_tokens=0,
            output_tokens=thoughts_token_count + candidates_token_count,
        )
        return usage

    # noinspection PyMethodMayBeStatic
    def native_response_to_assistant_message(self, response: google_types.GenerateContentResponse) -> AssistantMessage:
        """Convert Gemini response to AssistantMessage."""
        content = response.candidates[0].content
        assert content is not None
        blocks: list[AssistantTextBlock | ThinkingBlock | ToolCallBlock] = []
        tool_call_index = 0
        for native_block in content.parts:
            if native_block.text:
                if native_block.thought:
                    blocks.append(ThinkingBlock(text=native_block.text, signature=native_block.thought_signature))
                else:
                    blocks.append(AssistantTextBlock(text=native_block.text))
            elif native_block.function_call:
                blocks.append(
                    ToolCallBlock(
                        call_id=native_block.function_call.id,
                        name=native_block.function_call.name,
                        args=native_block.function_call.args,
                        index=tool_call_index,
                    )
                )
                tool_call_index += 1

        assistant_msg = AssistantMessage(content=blocks, usage=self.native_response_to_usage(response))
        return assistant_msg

    def native_response_to_structured_output(self, response: google_types.GenerateContentResponse) -> BaseModel | None:
        """Extract structured output from Gemini response."""
        if response.parsed is None:
            return None

        assert isinstance(response.parsed, BaseModel)
        return response.parsed

    async def count_input_tokens(self, messages: list[MessageType]) -> int:
        """Count input tokens for Gemini request."""
        response = await self._client.aio.models.count_tokens(
            model=self.model,
            contents=[self.system_prompt, *self.to_native_messages(messages)],  # Estimate
            config=google_types.CountTokensConfig(tools=self._config.tools),
        )
        return response.total_tokens

    def to_native_messages(self, messages: list[MessageType]) -> list[google_types.Content]:
        """Convert messages to Gemini format."""
        native_messages: list[google_types.Content] = []
        for msg in messages:
            native_parts: list[google_types.Part] = []

            if isinstance(msg, UserMessage):
                for user_block in msg.content:
                    if isinstance(user_block, UserTextBlock):
                        native_parts.append(google_types.Part.from_text(text=user_block.text))
                    elif isinstance(user_block, ToolResultBlock):
                        response_key = "error" if user_block.is_error else "result"
                        response = {response_key: user_block.result}
                        native_parts.append(
                            google_types.Part.from_function_response(
                                name=user_block.tool_call.name, response=response
                            )
                        )
                    else:
                        raise NotImplementedError

                native_messages.append(google_types.UserContent(parts=native_parts))

            elif isinstance(msg, AssistantMessage):
                for assistant_block in msg.content:
                    if isinstance(assistant_block, AssistantTextBlock):
                        native_parts.append(google_types.Part.from_text(text=assistant_block.text))
                    elif isinstance(assistant_block, ThinkingBlock):
                        native_parts.append(
                            google_types.Part(
                                text=assistant_block.text,
                                thought=True,
                                thought_signature=assistant_block.signature,
                            )
                        )
                    elif isinstance(assistant_block, ToolCallBlock):
                        native_parts.append(
                            google_types.Part.from_function_call(name=assistant_block.name, args=assistant_block.args)
                        )
                    else:
                        raise NotImplementedError

                native_messages.append(google_types.ModelContent(parts=native_parts))

            else:
                raise NotImplementedError

        return native_messages

    @retry(
        sleep=asyncio.sleep,
        wait=wait_fixed(10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(3),
    )
    async def generate_native_response(self, messages: list[MessageType]) -> google_types.GenerateContentResponse:
        """Generate response from Gemini with retry logic."""
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=self.to_native_messages(messages), config=self._config
        )

        print("Finish Reason:", response.candidates[0].finish_reason)

        return response


class AnthropicWrapper(
    ProviderWrapper[
        anthropic.AsyncAnthropic,
        anthropic_types_beta.BetaMessageParam,
        anthropic_types_beta.BetaMessage,
        anthropic_types_beta.BetaToolParam,
    ]
):
    """
    Anthropic Claude LLM wrapper.

    Note:
        Structured output not supported yet. Can enable support with
        tool_choice = {"type": "tool", "name": "Response"} in the future.
    """

    provider: Literal["anthropic"] = "anthropic"
    thinking_budget: int = 4096
    max_output_tokens: int = 64_000
    interleaved_thinking: bool = True
    base_uri: str | None = None
    api_key_env_var: str | None = None

    def setup_client(self) -> anthropic.AsyncAnthropic:
        """Set up Anthropic client."""
        return anthropic.AsyncAnthropic(
            base_url=self.base_uri, api_key=environ[self.api_key_env_var] if self.api_key_env_var else None
        )

    @staticmethod
    def to_native_tool_declaration(tool: type[BaseTool]) -> anthropic_types_beta.BetaToolParam:
        """Convert BaseTool to Anthropic tool declaration."""
        # noinspection PyTypeChecker
        return pydantic_to_function_declaration(tool, schema_key="input_schema")

    # noinspection PyMethodMayBeStatic
    def native_response_to_usage(self, response: anthropic_types_beta.BetaMessage) -> Usage:
        """Extract token usage from Anthropic response."""
        usage_data = response.usage
        usage = Usage(
            uncached_input_tokens=usage_data.input_tokens,
            cache_read_input_tokens=usage_data.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage_data.cache_creation_input_tokens or 0,
            output_tokens=usage_data.output_tokens,
        )
        return usage

    # noinspection PyMethodMayBeStatic
    def native_response_to_assistant_message(self, response: anthropic_types_beta.BetaMessage) -> AssistantMessage:
        """Convert Anthropic response to AssistantMessage."""
        blocks: list[AssistantTextBlock | ThinkingBlock | ToolCallBlock] = []
        tool_call_index = 0
        for native_block in response.content:
            if native_block.type == "text":
                blocks.append(AssistantTextBlock(text=native_block.text))
            elif native_block.type == "thinking":
                blocks.append(ThinkingBlock(text=native_block.thinking, signature=native_block.signature.encode()))
            elif native_block.type == "tool_use":
                assert isinstance(native_block.input, dict), f"args not a dict! Type: {type(native_block.input)}"
                blocks.append(
                    ToolCallBlock(
                        call_id=native_block.id,
                        name=native_block.name,
                        args=native_block.input,
                        index=tool_call_index,
                    )
                )
                tool_call_index += 1

        assistant_msg = AssistantMessage(content=blocks, usage=self.native_response_to_usage(response))
        return assistant_msg

    async def count_input_tokens(self, messages: list[MessageType]) -> int:
        """Count input tokens for Anthropic request."""
        response = await self._client.messages.count_tokens(
            model=self.model,
            thinking=(
                anthropic_types.ThinkingConfigEnabledParam(type="enabled", budget_tokens=self.thinking_budget)
                if self.thinking_budget
                else anthropic.NOT_GIVEN
            ),
            system=self.system_prompt,
            messages=self.to_native_messages(messages),
            tools=self._tools_declarations,
        )
        return response.input_tokens

    def to_native_messages(self, messages: list[MessageType]) -> list[anthropic_types_beta.BetaMessageParam]:
        """Convert messages to Anthropic format."""
        native_messages: list[anthropic_types_beta.BetaMessageParam] = []
        for msg in messages:
            native_parts: list[anthropic_types_beta.BetaContentBlockParam] = []

            if isinstance(msg, UserMessage):
                for user_block in msg.content:
                    if isinstance(user_block, UserTextBlock):
                        native_parts.append(anthropic_types_beta.BetaTextBlockParam(type="text", text=user_block.text))
                    elif isinstance(user_block, ToolResultBlock):
                        tool_result = anthropic_types_beta.BetaToolResultBlockParam(
                            type="tool_result", tool_use_id=user_block.tool_call.call_id, content=user_block.result
                        )
                        if user_block.is_error:
                            tool_result["is_error"] = True
                        native_parts.append(tool_result)
                    else:
                        raise NotImplementedError

                native_messages.append(anthropic_types_beta.BetaMessageParam(role="user", content=native_parts))

            elif isinstance(msg, AssistantMessage):
                for assistant_block in msg.content:
                    if isinstance(assistant_block, AssistantTextBlock):
                        native_parts.append(
                            anthropic_types_beta.BetaTextBlockParam(type="text", text=assistant_block.text)
                        )
                    elif isinstance(assistant_block, ThinkingBlock):
                        native_parts.append(
                            anthropic_types_beta.BetaThinkingBlockParam(
                                type="thinking",
                                thinking=assistant_block.text,
                                signature=assistant_block.signature.decode(),
                            )
                        )
                    elif isinstance(assistant_block, ToolCallBlock):
                        native_parts.append(
                            anthropic_types_beta.BetaToolUseBlockParam(
                                type="tool_use",
                                id=assistant_block.call_id,
                                name=assistant_block.name,
                                input=assistant_block.args,
                            )
                        )
                    else:
                        raise NotImplementedError

                native_messages.append(anthropic_types_beta.BetaMessageParam(role="assistant", content=native_parts))

            else:
                raise NotImplementedError

        return native_messages

    def get_native_objects_aligned_to_cache(
        self, messages: list[MessageType]
    ) -> tuple[list[BetaToolParam] | None, list[TextBlockParam] | None, list[BetaMessageParam]]:
        """
        Prepare objects with cache control markers for Anthropic prompt caching.

        Returns:
            Tuple of (tools_declarations, system_prompt, native_messages) with cache control
        """
        tools_declarations = self._tools_declarations
        system_prompt = (
            [anthropic_types.TextBlockParam(type="text", text=self.system_prompt)] if self.system_prompt else None
        )
        native_messages = self.to_native_messages(messages)

        if self.use_cache:
            if tools_declarations:
                tools_declarations = tools_declarations.copy()
                last_declaration = tools_declarations[-1].copy()
                last_declaration["cache_control"] = anthropic_types_beta.BetaCacheControlEphemeralParam(
                    type="ephemeral"
                )
                tools_declarations[-1] = last_declaration

            if system_prompt:
                system_prompt[-1]["cache_control"] = anthropic_types.CacheControlEphemeralParam(type="ephemeral")

            assert len(native_messages) > 0
            last_native_message = native_messages[-1]
            assert last_native_message["role"] == "user"
            last_content_block = last_native_message["content"]
            assert isinstance(last_content_block, list)
            last_part: anthropic_types_beta.BetaTextBlockParam | anthropic_types_beta.BetaToolResultBlockParam = (
                last_content_block[-1]
            )
            assert last_part["type"] in {"text", "tool_result"}
            last_part["cache_control"] = anthropic_types.CacheControlEphemeralParam(type="ephemeral")

        return tools_declarations, system_prompt, native_messages

    async def generate_native_response(self, messages: list[MessageType]):
        """Generate response from Anthropic with streaming and prompt caching."""
        max_tokens = self.max_output_tokens

        tools_declarations, system_prompt, native_messages = self.get_native_objects_aligned_to_cache(messages)

        async with self._client.beta.messages.stream(
            model=self.model,
            messages=native_messages,
            max_tokens=max_tokens,
            thinking=(
                anthropic_types.ThinkingConfigEnabledParam(type="enabled", budget_tokens=self.thinking_budget)
                if self.thinking_budget
                else anthropic.omit
            ),
            system=system_prompt if system_prompt else anthropic.omit,
            tools=tools_declarations if tools_declarations else anthropic.omit,
            timeout=600,
            betas=["interleaved-thinking-2025-05-14"] if self.interleaved_thinking else anthropic.omit,
        ) as stream:
            response = await stream.get_final_message()

        return response
