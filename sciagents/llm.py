from litellm import completion, acompletion
from typing import List, Dict, Generator, AsyncGenerator, Optional
from pydantic import BaseModel, ValidationError

class LlmConfig(BaseModel):
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
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
        self.model = config_dict.pop("model")
        self.config = config_dict

    def completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """
        Perform a non-streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.
            tools: Optional list of tool schemas in OpenAI format.

        Returns:
            Dict: Complete model response.

        Raises:
            ValueError: If model call fails.
        """
        try:
            kwargs = self.config.copy()
            if tools:
                kwargs["tools"] = tools
            response = completion(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )
            return response.choices[0].message
        except Exception as e:
            raise ValueError(f"LLM completion failed: {e}")

    def stream_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Generator:
        """
        Perform a streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.
            tools: Optional list of tool schemas in OpenAI format.

        Returns:
            Generator: Yields content (str) or tool call deltas (List[Dict]).

        Raises:
            ValueError: If model call fails.
        """
        try:
            kwargs = self.config.copy()
            if tools:
                kwargs["tools"] = tools
            response = completion(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                elif chunk.choices[0].delta.tool_calls:
                    yield chunk.choices[0].delta.tool_calls
        except Exception as e:
            raise ValueError(f"LLM stream completion failed: {e}")

    async def async_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """
        Perform an asynchronous non-streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.
            tools: Optional list of tool schemas in OpenAI format.

        Returns:
            Dict: Complete model response.

        Raises:
            ValueError: If model call fails.
        """
        try:
            kwargs = self.config.copy()
            if tools:
                kwargs["tools"] = tools
            response = await acompletion(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )
            return response.choices[0].message
        except Exception as e:
            raise ValueError(f"LLM async completion failed: {e}")

    async def async_stream_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> AsyncGenerator:
        """
        Perform an asynchronous streaming model call.

        Args:
            messages: List of message dictionaries in OpenAI format.
            tools: Optional list of tool schemas in OpenAI format.

        Returns:
            AsyncGenerator: Yields content (str) or tool call deltas (List[Dict]).

        Raises:
            ValueError: If model call fails.
        """
        try:
            kwargs = self.config.copy()
            if tools:
                kwargs["tools"] = tools
            response = await acompletion(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                elif chunk.choices[0].delta.tool_calls:
                    yield chunk.choices[0].delta.tool_calls
        except Exception as e:
            raise ValueError(f"LLM async stream completion failed: {e}")