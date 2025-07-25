import asyncio
import functools
from abc import ABC
from typing import Any, Callable, List, Literal, Optional, TypeVar

from .function_tool import FunctionTool
from mcp.server import FastMCP


class BaseToolkit(ABC):
    """Base class for tool collections.
    
    Provides core functionality for tool registration and MCP compatibility.
    Subclasses must implement the get_tools() method to provide the list of tools.
    """
    def __init__(self):

        self.mcp = FastMCP

    def get_tools(self) -> List[FunctionTool]:
        """Returns all tools in the toolkit.
        
        Returns:
            List of FunctionTool objects, each representing an available tool.
            
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement get_tools method")

    def run_mcp_server(self, mode: Literal["stdio", "sse", "streamable-http"]) -> None:
        r"""Run the MCP server in the specified mode.

        Args:
            mode (Literal["stdio", "sse", "streamable-http"]): The mode to run
                the MCP server in.
        """
        self.mcp.run(mode)