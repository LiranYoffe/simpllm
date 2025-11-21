"""Utility functions for simpllm package."""

from typing import Literal, Any, TypedDict, overload, Annotated

import jsonref
from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema


class BaseTool(BaseModel):
    """
    Base class for LLM tools.

    Tools should inherit from this class and implement the invoke() method.
    The __tool_name__ class attribute is automatically set from the class name
    or can be customized using the tool_name parameter.

    Example:
        class CalculatorTool(BaseTool):
            '''Perform arithmetic operations.'''
            operation: str = Field(description="Operation: add, subtract, multiply, divide")
            a: float = Field(description="First number")
            b: float = Field(description="Second number")

            async def invoke(self, ctx) -> str:
                ops = {"add": lambda x, y: x + y, ...}
                result = ops[self.operation](self.a, self.b)
                return f"Result: {result}"
    """

    def __init_subclass__(cls, *, tool_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__tool_name__ = cls.__name__ if tool_name is None else tool_name

    async def invoke(self, ctx) -> str:
        """
        Execute the tool.

        Args:
            ctx: Context object (type depends on the agent framework using simpllm)

        Returns:
            Tool execution result as string

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError


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

    # noinspection PyTypeChecker
    return {"name": tool_name, "description": schema.pop("description"), schema_key: schema}
