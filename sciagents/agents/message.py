from typing import List, Dict, Optional, Union, Any, Generator, AsyncGenerator
from pydantic import BaseModel, Field, ValidationError
from enum import Enum

class Role(str, Enum):
    """消息角色枚举，定义标准角色类型"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"  # 支持工具调用场景

class ToolCall(BaseModel):
    """工具调用结构"""
    id: str
    function: str
    arguments: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None  # 工具调用结果（可选）

class Message(BaseModel):
    """单条消息格式"""
    role: Role = Field(..., description="消息角色，如 user, assistant, system")
    content: str = Field(..., description="消息内容")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用列表（可选）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据（可选）")

    def to_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI 格式的字典"""
        result = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [tc.dict() for tc in self.tool_calls]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建 Message 实例"""
        try:
            role = Role(data.get("role"))
            content = data.get("content", "")
            tool_calls = [ToolCall(**tc) for tc in data.get("tool_calls", [])] if data.get("tool_calls") else None
            metadata = data.get("metadata")
            return cls(role=role, content=content, tool_calls=tool_calls, metadata=metadata)
        except Exception as e:
            raise ValidationError(f"Invalid message format: {e}")

class AgentInput(BaseModel):
    """Agent 输入格式"""
    messages: List[Message] = Field(..., description="消息列表")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文信息（可选）")

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI 格式的消息字典列表"""
        return [msg.to_dict() for msg in self.messages]

class AgentOutput(BaseModel):
    """Agent 输出格式"""
    content: Union[str, Generator, AsyncGenerator] = Field(..., description="输出内容（字符串或生成器）")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用结果（可选）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据（可选）")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_string(cls, content: str, tool_calls: Optional[List[ToolCall]] = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentOutput":
        """从字符串创建 AgentOutput"""
        return cls(content=content, tool_calls=tool_calls, metadata=metadata)

    @classmethod
    def from_generator(cls, generator: Union[Generator, AsyncGenerator], tool_calls: Optional[List[ToolCall]] = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentOutput":
        """从生成器创建 AgentOutput"""
        return cls(content=generator, tool_calls=tool_calls, metadata=metadata)