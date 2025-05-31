"""FunctionTool —— 把任意 Python 函数包装成 OpenAI 函数调用工具。"""
from __future__ import annotations

import inspect
import asyncio
import threading
from typing import Callable, Dict, List, Optional, Any, Type
from pydantic import BaseModel, create_model, ValidationError

class FunctionTool:
    """包装函数，自动生成 JSON-Schema，并负责参数校验、超时控制等。"""

    def __init__(
        self,
        func: Callable,
        *,
        openai_tool_schema: Optional[Dict] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or "No description provided."
        self.parameters = openai_tool_schema or self._extract_parameters()
        self.param_model = self._create_pyd_model()

    def _extract_parameters(self) -> Dict:
        sig = inspect.signature(self.func)
        props: Dict[str, Dict] = {}
        required: List[str] = []
        py2json = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
            list: "array",
            dict: "object",
        }
        for name, param in sig.parameters.items():
            if name in {"ctx", "context"}:
                continue
            anno = param.annotation if param.annotation != inspect.Parameter.empty else str
            jtype = py2json.get(anno, "string")
            props[name] = {"type": jtype}
            if param.default == inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": props, "required": required}

    def _create_pyd_model(self) -> Type[BaseModel] | None:
        fields = {}
        for k, v in self.parameters["properties"].items():
            typ = str
            match v["type"]:
                case "integer":
                    typ = int
                case "number":
                    typ = float
                case "boolean":
                    typ = bool
                case "array":
                    typ = list
                case "object":
                    typ = dict
            default_val = ... if k in self.parameters["required"] else None
            fields[k] = (typ, default_val)
        if not fields:
            return None
        return create_model(f"Params_{self.name}", **fields)  # type: ignore[arg-type]

    def get_function_name(self):
        return self.name

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }    
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the function with validated arguments, handling sync and async functions.
        Returns: {"result": Any} or {"error": str} if execution fails.
        """
        try:
            validated_args = self.param_model(**arguments).dict() if self.param_model else {}
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**validated_args)
            else:
                result = self.func(**validated_args)
            return {"result": result}
        except (ValidationError, TypeError, ValueError) as e:
            return {"error": f"Invalid arguments: {str(e)}"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    

# Function decorator
def function_tool(name_override: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to convert a function into a FunctionTool.
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name_override, description=description)
    return decorator
