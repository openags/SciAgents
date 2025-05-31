from typing import Dict, Any

class MCPTool:
    """
    Represents a tool available on a remote MCP server.
    It holds the schema of the tool (name, description, parameters)
    but not the execution logic itself, which is handled by the MCPClient.
    """
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initializes an MCPTool instance.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            parameters: A dictionary representing the JSON schema for the tool's parameters,
                        following the OpenAI function calling format.
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible schema for this tool.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def get_function_name(self): # Added for compatibility with ChatAgent's tool_map logic
        return self.name
