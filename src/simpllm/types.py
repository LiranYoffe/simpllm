"""Type definitions and unions for simpllm."""

from typing import Annotated

from pydantic import Field

from simpllm.wrappers import AnthropicWrapper, GeminiWrapper

# Provider wrapper union type
ProviderWrapperType = Annotated[GeminiWrapper | AnthropicWrapper, Field(discriminator="provider")]

__all__ = ["ProviderWrapperType"]
