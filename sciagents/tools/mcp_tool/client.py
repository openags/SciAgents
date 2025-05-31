import httpx
from typing import List, Dict, Any, Optional
from sciagents.tools.mcp_tool.toolkit import MCPTool
import json # For potential error parsing if response is not json

class MCPClient:
    """
    Client for interacting with an MCP (Message Control Protocol) server.
    It can list available tools and execute them remotely.

    Assumes the MCP server exposes HTTP endpoints for:
    - Listing tools (e.g., GET /tools)
    - Executing tools (e.g., POST /execute_tool)
    """

    def __init__(self, base_url: str):
        """
        Initializes the MCPClient.

        Args:
            base_url: The base URL of the MCP server (e.g., "http://localhost:8000").
        """
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url = base_url.rstrip('/')
        self._sync_client = httpx.Client()
        self._async_client = httpx.AsyncClient()

    def _parse_tool_list_response(self, response_json: Any) -> List[MCPTool]:
        if not isinstance(response_json, list):
            raise ValueError("Invalid response format for tool list: expected a list.")

        tools = []
        for tool_schema_wrapper in response_json:
            if not isinstance(tool_schema_wrapper, dict) or "function" not in tool_schema_wrapper:
                print(f"Skipping invalid tool schema wrapper: {tool_schema_wrapper}")
                continue

            tool_schema = tool_schema_wrapper.get("function", {})
            if not tool_schema.get("name") or not tool_schema.get("description"):
                print(f"Skipping tool with missing name or description: {tool_schema}")
                continue

            tools.append(
                MCPTool(
                    name=tool_schema["name"],
                    description=tool_schema["description"],
                    parameters=tool_schema.get("parameters", {"type": "object", "properties": {}}),
                )
            )
        return tools

    def list_tools(self) -> List[MCPTool]:
        """
        Fetches the list of available tools from the MCP server.

        Returns:
            A list of MCPTool instances.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
            ValueError: If the response format is invalid.
        """
        try:
            response = self._sync_client.get(f"{self.base_url}/tools")
            response.raise_for_status()  # Raise an exception for bad status codes
            return self._parse_tool_list_response(response.json())
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to MCP server at {self.base_url}/tools: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response from {self.base_url}/tools: {e}")


    async def a_list_tools(self) -> List[MCPTool]:
        """
        Asynchronously fetches the list of available tools from the MCP server.

        Returns:
            A list of MCPTool instances.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
            ValueError: If the response format is invalid.
        """
        try:
            response = await self._async_client.get(f"{self.base_url}/tools")
            response.raise_for_status()
            return self._parse_tool_list_response(response.json())
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to MCP server at {self.base_url}/tools: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response from {self.base_url}/tools: {e}")

    def _parse_execute_response(self, response_json: Any) -> Any:
        if not isinstance(response_json, dict):
            raise ValueError("Invalid response format for tool execution: expected a dictionary.")

        if "error" in response_json:
            raise RuntimeError(f"MCP tool execution failed: {response_json['error']}")
        if "result" not in response_json:
            # Fallback if 'result' is not present but no 'error' either
            # This depends on the server's response contract
            # For now, we assume 'result' should be there on success
            raise ValueError("Invalid response format for tool execution: 'result' key missing.")
        return response_json["result"]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Executes a tool remotely on the MCP server.

        Args:
            tool_name: The name of the tool to execute.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The result of the tool execution.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
            RuntimeError: If the tool execution on the server side returns an error.
            ValueError: If the response format is invalid.
        """
        payload = {"tool_name": tool_name, "arguments": arguments}
        try:
            response = self._sync_client.post(f"{self.base_url}/execute", json=payload) # Changed to /execute
            response.raise_for_status()
            return self._parse_execute_response(response.json())
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to MCP server at {self.base_url}/execute: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response from {self.base_url}/execute: {e}")


    async def a_execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Asynchronously executes a tool remotely on the MCP server.

        Args:
            tool_name: The name of the tool to execute.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The result of the tool execution.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
            RuntimeError: If the tool execution on the server side returns an error.
            ValueError: If the response format is invalid.
        """
        payload = {"tool_name": tool_name, "arguments": arguments}
        try:
            response = await self._async_client.post(f"{self.base_url}/execute", json=payload) # Changed to /execute
            response.raise_for_status()
            return self._parse_execute_response(response.json())
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to MCP server at {self.base_url}/execute: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response from {self.base_url}/execute: {e}")

    def close(self):
        """Closes the underlying HTTP clients. Should be called when the client is no longer needed."""
        self._sync_client.close()
        # For async client, it's better to use 'async with' block or call aclose manually
        # httpx.AsyncClient.aclose() is an async method.
        # This simple close method might not be sufficient if used heavily outside 'async with'.

    async def aclose(self):
        """Asynchronously closes the underlying HTTP clients."""
        await self._async_client.aclose()

# Example of how the server might provide /tools (conceptual)
# This would be part of the MCPServer/FastMCP implementation
# GET /tools response:
# [
#   {
#     "type": "function",
#     "function": {
#       "name": "get_weather",
#       "description": "Get the current weather in a given location",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
#         },
#         "required": ["location"]
#       }
#     }
#   }
# ]

# Example of how the server might provide /execute (conceptual)
# POST /execute request body:
# {
#   "tool_name": "get_weather",
#   "arguments": {"location": "Tokyo"}
# }
# POST /execute response body (success):
# {
#   "result": {"temperature": "15 C", "condition": "Cloudy"}
# }
# POST /execute response body (error):
# {
#   "error": "Invalid location format"
# }
