"""Utility functions for simpllm package."""

from typing import Any, Literal, TypedDict, overload

import jsonref  # type: ignore[import-untyped]
from pydantic import ConfigDict
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from simpllm.tools import BaseTool


class GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    """JSON schema generator that omits titles."""

    def field_title_should_be_set(self, schema: core_schema.CoreSchema) -> bool:
        return False

    def _update_class_schema(self, json_schema: JsonSchemaValue, cls: type[Any], config: ConfigDict) -> None:
        super()._update_class_schema(json_schema, cls, config)
        if isinstance(json_schema, dict):
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
    tool: type[BaseTool], schema_key: Literal["parameters"] = ...
) -> ToolDeclaration: ...


@overload
def pydantic_to_function_declaration(
    tool: type[BaseTool], schema_key: Literal["input_schema"] = ...
) -> ToolDeclarationAnthropic: ...


def pydantic_to_function_declaration(
    tool: type[BaseTool], schema_key: Literal["parameters", "input_schema"] = "parameters"
) -> ToolDeclaration | ToolDeclarationAnthropic:
    """
    Convert BaseTool subclass to LLM function declaration.

    The tool's __tool_name__ class attribute is used for the function name.
    The model's description is extracted from the docstring/description field.

    Support tested for simple types, Path, Literal, dict[str, str].
    Enum/StrEnum not supported. Union and list - not tested!

    Args:
        tool: BaseTool subclass representing the tool
        schema_key: Either "parameters" (Gemini) or "input_schema" (Anthropic)

    Returns:
        Tool declaration dict for the specified provider
    """
    schema = tool.model_json_schema(schema_generator=GenerateJsonSchemaNoTitles)

    if "$defs" in schema:
        schema = jsonref.replace_refs(schema, proxies=False)
        schema.pop("$defs")

    # Get tool name from __tool_name__ attribute (set by BaseTool.__init_subclass__)
    tool_name = tool.__tool_name__
    description = str(schema.pop("description"))

    if schema_key == "parameters":
        return ToolDeclaration(name=tool_name, description=description, parameters=schema)
    else:
        return ToolDeclarationAnthropic(name=tool_name, description=description, input_schema=schema)
