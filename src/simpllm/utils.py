"""Utility functions for simpllm package."""

import logging
import sys
from typing import Literal, Any, TypedDict, overload

import jsonref
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema


def get_logger(name: str):
    """
    Sets up a Python logger to output messages at all logging levels
    to the console using a detailed, explicit configuration.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if the function is called multiple times
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)  # Logger processes all levels
    return logger


class GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    """JSON schema generator that omits titles."""

    def field_title_should_be_set(self, _) -> bool:
        return False

    def _update_class_schema(self, json_schema, cls, config) -> None:
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)


class ToolDeclaration(TypedDict):
    """Tool declaration for Google Gemini."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDeclarationAnthropic(TypedDict):
    """Tool declaration for Anthropic Claude."""

    name: str
    description: str
    input_schema: dict[str, Any]


@overload
def pydantic_to_function_declaration(
    tool: type[BaseModel], schema_key: Literal["parameters"] = ...
) -> ToolDeclaration: ...


@overload
def pydantic_to_function_declaration(
    tool: type[BaseModel], schema_key: Literal["input_schema"] = ...
) -> ToolDeclarationAnthropic: ...


def pydantic_to_function_declaration(
    tool: type[BaseModel], schema_key: Literal["parameters", "input_schema"] = "parameters"
) -> ToolDeclaration | ToolDeclarationAnthropic:
    """
    Convert Pydantic model to LLM function declaration.

    Expects the tool model to have a __tool_name__ class attribute for the function name.
    The model's description is extracted from the docstring/description field.

    Support tested for simple types, Path, Literal, dict[str, str].
    Enum/StrEnum not supported. Union and list - not tested!

    Args:
        tool: Pydantic model representing the tool
        schema_key: Either "parameters" (Gemini) or "input_schema" (Anthropic)

    Returns:
        Tool declaration dict for the specified provider
    """
    schema = tool.model_json_schema(schema_generator=GenerateJsonSchemaNoTitles)

    if "$defs" in schema:
        schema = jsonref.replace_refs(schema, proxies=False)
        schema.pop("$defs")

    # Get tool name from __tool_name__ attribute if available, otherwise use class name
    tool_name = getattr(tool, "__tool_name__", tool.__name__)

    # noinspection PyTypeChecker
    return {"name": tool_name, "description": schema.pop("description"), schema_key: schema}
