from __future__ import annotations

import inspect
import json
import os
import shlex
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    cast,
)
from urllib.parse import urlparse

from ..base_tool import BaseToolkit
from ..function_tool import FunctionTool

# SciAgents provides run_async (can be implemented by user), here is a simple version
import asyncio

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from mcp import ClientSession, Tool, ListToolsResult

import logging
logger = logging.getLogger("sciagents.mcpclient")


class MCPClient(BaseToolkit):
    """Establishes a connection to a single MCP Server and dynamically generates FunctionTool instances."""

    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        mode: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__(timeout=timeout or 10.0)

        # Save parameters
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.headers = headers or {}
        self.mode = mode
        self.strict = strict

        # Runtime attributes
        self._mcp_tools: List["Tool"] = []
        self._session: Optional["ClientSession"] = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

    # ------------------------------------------------------------------
    async def connect(self):
        """Establish connection and retrieve remote tool list."""
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        if self._connected:
            logger.warning("Already connected")
            return self

        try:
            # 1) Select channel
            if urlparse(self.command_or_url).scheme in ("http", "https"):
                if self.mode in (None, "sse"):
                    read_stream, write_stream = await self._exit_stack.enter_async_context(
                        sse_client(self.command_or_url, headers=self.headers, timeout=self.timeout)
                    )
                elif self.mode == "streamable-http":
                    read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                        streamablehttp_client(
                            self.command_or_url,
                            headers=self.headers,
                            timeout=timedelta(seconds=self.timeout) if self.timeout else None,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported mode {self.mode} for HTTP url")
            else:
                # stdio local executable
                command = self.command_or_url
                arguments = self.args
                if not arguments:
                    argv = shlex.split(command)
                    command, arguments = argv[0], argv[1:]
                if os.name == "nt" and command.lower() == "npx":
                    command = "npx.cmd"
                server_params = StdioServerParameters(command=command, args=arguments, env={**os.environ, **self.env})
                read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(server_params))

            # 2) Create session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream, timedelta(seconds=self.timeout) if self.timeout else None)
            )
            await self._session.initialize()

            # 3) Fetch remote tools
            list_tools_result = await self._session.list_tools()
            self._mcp_tools = list_tools_result.tools  # type: ignore[attr-defined]
            self._connected = True
            return self
        except Exception as e:
            await self.disconnect()
            logger.error(f"Connect MCP failed: {e}")
            raise e

    def connect_sync(self):
        return run_async(self.connect())

    # ------------------------------------------------------------------
    async def disconnect(self):
        if not self._connected:
            return
        self._connected = False
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._session = None

    # ------------------------------------------------------------------
    @asynccontextmanager
    async def connection(self):
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    # ------------------------------------------------------------------
    # Convert remote Tool description to dynamic async function
    # ------------------------------------------------------------------
    def generate_function_from_mcp_tool(self, mcp_tool: "Tool") -> Callable:
        func_name = mcp_tool.name
        func_desc = mcp_tool.description or "No description provided."
        schema_props = mcp_tool.inputSchema.get("properties", {})
        required_params = mcp_tool.inputSchema.get("required", [])

        # Build Python type hints
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        annotations: Dict[str, Any] = {}
        defaults: Dict[str, Any] = {}
        for pname, p_schema in schema_props.items():
            annotations[pname] = type_map.get(p_schema.get("type", "string"), str)
            if pname not in required_params:
                defaults[pname] = None

        async def _dynamic(**kwargs) -> str:
            from mcp.types import CallToolResult  # import requires mcp installed

            missing = set(required_params) - kwargs.keys()
            if missing:
                return f"Missing required params: {missing}"
            if not self._session:
                raise RuntimeError("MCPClient not connected")
            result: CallToolResult = await self._session.call_tool(func_name, kwargs)
            if not result.content:
                return "<empty>"
            content = result.content[0]
            return getattr(content, "text", str(content))

        _dynamic.__name__ = func_name
        _dynamic.__doc__ = func_desc
        _dynamic.__annotations__ = annotations
        params = [
            inspect.Parameter(
                name=p,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=defaults.get(p, inspect.Parameter.empty),
                annotation=annotations[p],
            )
            for p in schema_props.keys()
        ]
        _dynamic.__signature__ = inspect.Signature(parameters=params)  # type: ignore[attr-defined]
        return _dynamic

    # ------------------------------------------------------------------
    def _build_tool_schema(self, mcp_tool: "Tool") -> Dict[str, Any]:
        input_schema = mcp_tool.inputSchema
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "No description provided.",
                "parameters": input_schema,
                "strict": self.strict,
            },
        }

    # ------------------------------------------------------------------
    def get_tools(self):
        return [
            FunctionTool(
                self.generate_function_from_mcp_tool(t),
                openai_tool_schema=self._build_tool_schema(t),
            )
            for t in self._mcp_tools
        ]
