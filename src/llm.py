from litellm import completion, acompletion
from typing import List, Dict, Generator, AsyncGenerator, Optional
from pydantic import BaseModel, ValidationError

class LlmConfig(BaseModel):
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    # 其他默认参数（例如 temperature、max_tokens 等）也可以在这里定义
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class LlmModel:
    """
    Client for interacting with LLMs using litellm.
    """

    def __init__(self, config: LlmConfig) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LlmConfig object with model name and optional parameters.
        """
        config_dict = config.dict(exclude_unset=True)
        self.model = config_dict.pop("model")  # 单独取出 model
        self.config = config_dict   

    def completion(self, messages: List[Dict]) -> str:
        """
        Perform a non-streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            str: Complete model response.

        Raises:
            ValueError: If model call fails.
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                stream=False,
                **self.config
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"LLM completion failed: {e}")

    def stream_completion(self, messages: List[Dict]) -> Generator:
        """
        Perform a streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            Generator: Yields content chunks as they are received.

        Raises:
            ValueError: If model call fails.
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                stream=True,
                **self.config
            )
            for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                yield content
        except Exception as e:
            raise ValueError(f"LLM stream completion failed: {e}")

    async def async_completion(self, messages: List[Dict]) -> str:
        """
        Perform an asynchronous non-streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            str: Complete model response.

        Raises:
            ValueError: If model call fails.
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                stream=False,
                **self.config
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"LLM async completion failed: {e}")

    async def async_stream_completion(self, messages: List[Dict]) -> AsyncGenerator:
        """
        Perform an asynchronous streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            AsyncGenerator: Yields content chunks as they are received.

        Raises:
            ValueError: If model call fails.
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                stream=True,
                **self.config
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                yield content
        except Exception as e:
            raise ValueError(f"LLM async stream completion failed: {e}")