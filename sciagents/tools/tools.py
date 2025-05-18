from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model, ValidationError
from inspect import signature, getdoc
import json
import asyncio

class FunctionTool:
    """
    Wraps a function as a tool, providing OpenAI-compatible schema with parameter validation.
    """

    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize the FunctionTool.

        Args:
            func: The function to wrap.
            name: Optional name for the tool (defaults to function name).
            description: Optional description (defaults to function docstring).
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or (getdoc(func) or "")
        self.parameters = self._extract_parameters()
        self.param_model = self._create_param_model()

    def _extract_parameters(self) -> Dict[str, Any]:
        """
        Extract function parameters as JSON schema.
        """
        sig = signature(self.func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in {"ctx", "context"}:
                continue
            param_type = param.annotation if param.annotation != param.empty else str
            properties[param_name] = {"type": self._type_to_schema(param_type)}
            if param.default == param.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _type_to_schema(self, param_type: Any) -> str:
        """
        Convert Python type to JSON schema type.
        """
        if param_type in {int, float}:
            return "number"
        elif param_type == str:
            return "string"
        elif param_type == bool:
            return "boolean"
        elif param_type in {list, List}:
            return "array"
        elif param_type in {dict, Dict}:
            return "object"
        return "string"

    def _create_param_model(self) -> Type[BaseModel]:
        """
        Create a Pydantic model for parameter validation.
        """
        sig = signature(self.func)
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in {"ctx", "context"}:
                continue
            param_type = param.annotation if param.annotation != param.empty else str
            fields[param_name] = (param_type, ... if param.default == param.empty else param.default)
        return create_model(f"{self.name}_Params", **fields)

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI-compatible tool schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the function with validated arguments, handling sync and async functions.

        Args:
            arguments: Dictionary of function arguments.

        Returns:
            Dict: {"result": Any} or {"error": str} if execution fails.
        """
        try:
            # Validate parameters
            validated_args = self.param_model(**arguments).dict()
            # Execute function
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**validated_args)
            else:
                result = self.func(**validated_args)
            return {"result": result}
        except (ValidationError, TypeError, ValueError) as e:
            return {"error": f"Invalid arguments: {str(e)}"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

def function_tool(name_override: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to convert a function into a FunctionTool.
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name_override, description=description)
    return decorator