from typing import List, Dict, Optional, Union, Any, Generator, AsyncGenerator
from pydantic import BaseModel, Field, ValidationError
from enum import Enum

class Role(str, Enum):
    """Role enumeration for message types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"  # For tool call scenarios

class ToolCall(BaseModel):
    """Tool call structure, function field supports dict or str (OpenAI compatible)."""
    id: str
    function: Union[str, Dict[str, Any]]  # Compatible with OpenAI format
    arguments: Optional[Dict[str, Any]] = None  # Optional, not required if function is dict
    result: Optional[Dict[str, Any]] = None  # Optional tool call result

class Message(BaseModel):
    """Single message format, tool_calls compatible with OpenAI format."""
    role: Role = Field(..., description="Message role, e.g. user, assistant, system")
    content: str = Field(..., description="Message content")
    name : Optional[str] = Field(None, description="Name of the assistant (optional)")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID (optional, for tool messages)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (optional)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible dict. Function field is pass-through. For tool messages, auto-add tool_call_id."""
        result = {"role": self.role.value, "name": self.name, "content": self.content}

        if self.role == Role.TOOL:
            result["tool_call_id"] = self.tool_call_id

        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message instance from a dict, compatible with function as dict."""
        try:
            role = Role(data.get("role"))
            content = data.get("content", "")

            if role == Role.TOOL:
                 tool_call_id = data.get("tool_call_id")

            if data.get("metadata"):
                metadata = data.get("metadata")

            return cls(role=role, content=content, tool_call_id=tool_call_id, metadata=metadata)
        
        except Exception as e:
            raise ValidationError(f"Invalid message format: {e}")

class AgentInput(BaseModel):
    """Agent input format."""
    messages: List[Message] = Field(..., description="List of messages")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (optional)")

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to a list of OpenAI-compatible message dicts."""
        return [msg.to_dict() for msg in self.messages]

class AgentOutput(BaseModel):
    """Agent output format."""
    content: Union[str, Generator, AsyncGenerator] = Field(..., description="Output content (string or generator)")
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = Field(None, description="Tool call results (optional)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (optional)")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_string(cls, content: str, tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentOutput":
        """Create AgentOutput from a string."""
        return cls(content=content, tool_calls=tool_calls, metadata=metadata)

    @classmethod
    def from_generator(cls, generator: Union[Generator, AsyncGenerator], tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentOutput":
        """Create AgentOutput from a generator."""
        return cls(content=generator, tool_calls=tool_calls, metadata=metadata)