from typing import List, Dict, Union, Any, Generator, AsyncGenerator
from src.agents.agent import Agent
from src.agents.message import AgentInput, AgentOutput, Message
from src.llm import LlmModel, LlmConfig

class ChatAgent(Agent):
    """
    A simple chat agent that maintains conversation history and supports streaming.
    Uses LlmModel from llm.py for LLM interactions.
    """

    def __init__(self, name: str, llm_config: Union[LlmConfig, Dict[str, Any]], stream: bool = False) -> None:
        """
        Initialize the chat agent.

        Args:
            name: Name of the agent.
            llm_config: LlmConfig instance or dict with model configuration (passed to LlmModel).
            stream: Whether to enable streaming output, defaults to False.
        """
        super().__init__(name, llm_config, stream)
        self.history: List[Dict] = []  # 存储 OpenAI 格式的对话历史

    def run(self, input_data: Union[AgentInput, List[Dict]], *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Run the agent with the given input, equivalent to a single step.

        Args:
            input_data: Input messages as AgentInput or list of dictionaries.
            *args, **kwargs: Additional arguments for flexibility.

        Returns:
            AgentOutput: Response content and optional metadata.
        """
        return self.step(input_data, *args, **kwargs)

    def reset(self) -> None:
        """
        Reset the agent's conversation history.
        """
        self.history = []

    def step(self, input_data: Union[AgentInput, List[Dict]], *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step of the agent's operation, adding input to history.

        Args:
            input_data: Input messages as AgentInput or list of dictionaries.
            *args, **kwargs: Additional arguments for flexibility.

        Returns:
            AgentOutput: Contains response content (str or Generator) and optional metadata.

        Uses self.llm_model (LlmModel from llm.py) for completion or stream_completion.
        """
        # 转换输入为 OpenAI 格式
        messages = input_data.to_dict_list() if isinstance(input_data, AgentInput) else input_data
        self.history.extend(messages)

        # 使用 LlmModel 进行模型调用
        if self.stream:
            return AgentOutput.from_generator(
                self.llm_model.stream_completion(self.history)  # 调用 llm.py 的 stream_completion
            )
        content = self.llm_model.completion(self.history)  # 调用 llm.py 的 completion
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)

    async def a_step(self, input_data: Union[AgentInput, List[Dict]], *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step of the agent's operation asynchronously.

        Args:
            input_data: Input messages as AgentInput or list of dictionaries.
            *args, **kwargs: Additional arguments for flexibility.

        Returns:
            AgentOutput: Contains response content (str or AsyncGenerator) and optional metadata.

        Uses self.llm_model (LlmModel from llm.py) for async_completion or async_stream_completion.
        """
        # 转换输入为 OpenAI 格式
        messages = input_data.to_dict_list() if isinstance(input_data, AgentInput) else input_data
        self.history.extend(messages)

        # 使用 LlmModel 进行异步模型调用
        if self.stream:
            return AgentOutput.from_generator(
                self.llm_model.async_stream_completion(self.history)  # 调用 llm.py 的 async_stream_completion
            )
        content = await self.llm_model.async_completion(self.history)  # 调用 llm.py 的 async_completion
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)