import json
from typing import Callable, List, Dict, Optional, Union, Any, Generator, AsyncGenerator
from sciagents.agents.agent import Agent
from sciagents.agents.message import AgentInput, AgentOutput, Message
from sciagents.llm import LlmModel, LlmConfig
from sciagents.tools import FunctionTool, function_tool
import asyncio
import time

class ChatAgent(Agent):
    """
    A simple chat agent that maintains conversation history and supports streaming.
    Uses LlmModel from llm.py for LLM interactions.
    """

    def __init__(self, name: str, llm_config: Union[LlmConfig, Dict[str, Any]], tools: Optional[List[Union[Callable, FunctionTool]]] = None) -> None:
        """
        Initialize the chat agent.

        Args:
            name: Name of the agent.
            llm_config: LlmConfig instance or dict with model configuration.
            tools: List of functions or FunctionTool instances.
        """
        super().__init__(name, llm_config)
        self.history: List[Dict] = []
        self.tools: List[FunctionTool] = []
        self.tool_map: Dict[str, FunctionTool] = {}
        if tools:
            for tool in tools:
                if isinstance(tool, FunctionTool):
                    self.tools.append(tool)
                else:
                    self.tools.append(FunctionTool(tool))
                self.tool_map[tool.name if isinstance(tool, FunctionTool) else tool.__name__] = self.tools[-1]

    def run(self, input_data: Union[AgentInput, List[Dict]], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Run the agent with the given input, equivalent to a single step.
        """
        return self.step(input_data, *args, **kwargs)

    def reset(self) -> None:
        """
        Reset the agent's conversation history.
        """
        self.history = []

    def step(self, input_data: Union[AgentInput, List[Dict], str], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step, handling tool calls and streaming.
        """
        if isinstance(input_data, str):
            # Convert string to proper message format
            messages = [{
                "role": "user",
                "content": input_data,
            }]
        elif isinstance(input_data, AgentInput):
            messages = input_data.to_dict_list()
        else:
            messages = input_data        
            
        self.history.extend(messages)

        tool_schemas = [tool.to_openai_schema() for tool in self.tools] if self.tools else None
        if stream:
            generator = self.llm_model.stream_completion(self.history, tools=tool_schemas)
            return self._handle_streaming(generator)
        else:
            response = self.llm_model.completion(self.history, tools=tool_schemas)
            return self._handle_response(response)

    async def a_step(self, input_data: Union[AgentInput, List[Dict], str], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        """
        Perform a single step asynchronously, handling tool calls and streaming.
        """
        if isinstance(input_data, str):
            # Convert string to proper message format
            messages = [{
                "role": "user",
                "content": input_data,
            }]
        elif isinstance(input_data, AgentInput):
            messages = input_data.to_dict_list()
        else:
            messages = input_data        

        self.history.extend(messages)

        tool_schemas = [tool.to_openai_schema() for tool in self.tools] if self.tools else None
        if stream:
            generator = self.llm_model.async_stream_completion(self.history, tools=tool_schemas)
            return await self._handle_async_streaming(generator)
        else:
            response = await self.llm_model.async_completion(self.history, tools=tool_schemas)
            return await self._handle_async_response(response)

    def _handle_streaming(self, generator: Generator) -> AgentOutput:
        """
        Handle streaming responses, maintaining generator output for content and tool calls.
        """
        def combined_generator() -> Generator[str, None, None]:
            collected_content = []
            final_tool_calls = {}  # Dictionary to accumulate tool calls by index
            for chunk in generator:
                if isinstance(chunk, str):
                    collected_content.append(chunk)
                    yield chunk
                elif isinstance(chunk, list):
                    for tool_call in chunk:
                        index = getattr(tool_call, 'index', 0)
                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call
                        else:
                            final_tool_calls[index].function.arguments += tool_call.function.arguments

            if final_tool_calls:
                yield "[Executing tools...]\n"
                tool_call_results = self._execute_tool_calls(list(final_tool_calls.values()))
                self.history.append({"role": "assistant", "tool_calls": list(final_tool_calls.values())})
                self.history.extend(tool_call_results)

                final_generator = self.llm_model.stream_completion(self.history)
                final_content = []
                for chunk in final_generator:
                    if isinstance(chunk, str):
                        final_content.append(chunk)
                        yield chunk
                    elif isinstance(chunk, list):
                        error_msg = json.dumps({"error": "Multiple tool calls not supported in single step"}) + "\n"
                        yield error_msg
                        self.history.append({"role": "assistant", "content": error_msg.strip()})
                        return
                if final_content:
                    content = "".join(final_content)
                    self.history.append({"role": "assistant", "content": content})


            elif collected_content:
                content = "".join(collected_content)
                self.history.append({"role": "assistant", "content": content})
            else:
                content = "no content"
                self.history.append({"role": "assistant", "content": content})

        return AgentOutput.from_generator(combined_generator())

    async def _handle_async_streaming(self, generator: AsyncGenerator) -> AgentOutput:
        """
        Handle async streaming responses, maintaining async generator output.
        """
        async def combined_generator() -> AsyncGenerator[str, None]:
            collected_content = []
            final_tool_calls = {}  # Dictionary to accumulate tool calls by index
            async for chunk in generator:
                if isinstance(chunk, str):
                    collected_content.append(chunk)
                    yield chunk
                elif isinstance(chunk, list):
                    for tool_call in chunk:
                        index = getattr(tool_call, 'index', 0)
                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call
                        else:
                            final_tool_calls[index].function.arguments += tool_call.function.arguments

            if final_tool_calls:

                yield "[Executing tools...]\n"
                tool_call_results = await self._aexecute_tool_calls(list(final_tool_calls.values()))
                self.history.append({"role": "assistant", "tool_calls": list(final_tool_calls.values())})
                self.history.extend(tool_call_results)


                final_generator = self.llm_model.async_stream_completion(self.history)
                final_content = []
                async for chunk in final_generator:
                    if isinstance(chunk, str):
                        final_content.append(chunk)
                        yield chunk
                    elif isinstance(chunk, list):
                        error_msg = json.dumps({"error": "Multiple tool calls not supported in single step"}) + "\n"
                        yield error_msg
                        self.history.append({"role": "assistant", "content": error_msg.strip()})
                        return
                if final_content:
                    content = "".join(final_content)
                    self.history.append({"role": "assistant", "content": content})


            elif collected_content:

                content = "".join(collected_content)
                self.history.append({"role": "assistant", "content": content})
            else:
                content = "no content"
                self.history.append({"role": "assistant", "content": content})

        return AgentOutput.from_generator(combined_generator())

    def _handle_response(self, response: Dict) -> AgentOutput:
        """
        Handle non-streaming responses synchronously.
        """
        if response.get("tool_calls"):
            self.history.append({"role": "assistant", "tool_calls": response["tool_calls"]})
            tool_call_results = self._execute_tool_calls(response["tool_calls"])
            self.history.extend(tool_call_results)
            final_response = self.llm_model.completion(self.history)
            if final_response.get("tool_calls"):
                content = json.dumps({"error": "Multiple tool calls not supported in single step"})
                self.history.append({"role": "assistant", "content": content})
                return AgentOutput.from_string(content)
            content = final_response.get("content", "")
            self.history.append({"role": "assistant", "content": content})
            return AgentOutput.from_string(content)
        content = response.get("content", "")
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)

    async def _handle_async_response(self, response: Dict) -> AgentOutput:
        """
        Handle non-streaming responses asynchronously.
        """
        if response.get("tool_calls"):
            self.history.append({"role": "assistant", "tool_calls": response["tool_calls"]})
            tool_call_results = await self._aexecute_tool_calls(response["tool_calls"])
            self.history.extend(tool_call_results)
            final_response = await self.llm_model.async_completion(self.history)
            if final_response.get("tool_calls"):
                content = json.dumps({"error": "Multiple tool calls not supported in single step"})
                self.history.append({"role": "assistant", "content": content})
                return AgentOutput.from_string(content)
            content = final_response.get("content", "")
            self.history.append({"role": "assistant", "content": content})
            return AgentOutput.from_string(content)
        content = response.get("content", "")
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)

    def _execute_tool_calls(self, tool_calls: List[Any]) -> List[Dict]:
        """
        Execute tool calls synchronously with parameter validation and error handling.
        Aggregates fragmented tool call arguments.
        """
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                arguments = json.loads(arguments_str) if arguments_str else {}
                tool = self.tool_map.get(tool_name)
                if not tool:
                    results.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Tool {tool_name} not found",
                        "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                    })
                    continue
                start_time = time.time()
                # Use sync execution for sync tools, async for async tools
                if asyncio.iscoroutinefunction(tool.func):
                    # Create a new event loop for async tools
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(asyncio.wait_for(tool.execute(arguments), timeout=10.0))
                    finally:
                        loop.close()
                else:
                    result = tool.func(**arguments)
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except json.JSONDecodeError:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Invalid arguments JSON: {arguments_str}",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except asyncio.TimeoutError:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Tool execution timed out after 10 seconds",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Tool execution failed: {str(e)}",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
        return results

    async def _aexecute_tool_calls(self, tool_calls: List[Any]) -> List[Dict]:
        """
        Execute tool calls asynchronously with parameter validation and error handling.
        Aggregates fragmented tool call arguments.
        """
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                arguments = json.loads(arguments_str) if arguments_str else {}
                tool = self.tool_map.get(tool_name)
                if not tool:
                    results.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Tool {tool_name} not found",
                        "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                    })
                    continue
                start_time = time.time()
                result = await asyncio.wait_for(tool.execute(arguments), timeout=10.0)
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except json.JSONDecodeError:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Invalid arguments JSON: {arguments_str}",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except asyncio.TimeoutError:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Tool execution timed out after 10 seconds",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Tool execution failed: {str(e)}",
                    "tool_call_id": tool_call.id or f"unknown_{id(tool_call)}"
                })
        return results