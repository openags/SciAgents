from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator, Awaitable
from abc import ABC, abstractmethod
from sciagents.llm import LlmModel, LlmConfig
from sciagents.agents.message import AgentInput, AgentOutput

class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, name: str, llm_config: Union[LlmConfig, Dict[str, Any]]) -> None:
        """
        Initialize the agent with a name and LLM configuration.

        Args:
            name: Name of the agent.
            llm_config: Either an LlmConfig instance or a dict containing configuration parameters
                       (including 'model', 'api_key', etc.). See LlmConfig for details.
        """
        self.name = name
        if isinstance(llm_config, dict):
            try:
                self.llm_config = LlmConfig(**llm_config)
            except Exception as e:
                raise ValueError(f"Invalid llm_config: {e}")
        else:
            self.llm_config = llm_config
        self.llm_model = LlmModel(config=self.llm_config)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the agent with the given arguments.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """
        pass

    @abstractmethod
    def step(self, input_data: Union[AgentInput, List[Dict], str], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step of the agent's operation.

        Args:
            input_data: Input messages as AgentInput or list of dictionaries.
            stream: Whether to enable streaming output, defaults to False.
            *args, **kwargs: Additional arguments for flexibility.

        Returns:
            AgentOutput: Contains response content (str or Generator) and optional metadata.
        """
        pass

    @abstractmethod
    async def a_step(self, input_data: Union[AgentInput, List[Dict], str], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step of the agent's operation asynchronously.

        Args:
            input_data: Input messages as AgentInput or list of dictionaries.
            stream: Whether to enable streaming output, defaults to False.
            *args, **kwargs: Additional arguments for flexibility.

        Returns:
            AgentOutput: Contains response content (str or AsyncGenerator) and optional metadata.
        """
        pass