from pydantic import BaseModel


class BaseTool[ContextType](BaseModel):
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

    async def invoke(self, ctx: ContextType) -> str:
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
