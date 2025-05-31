import json
from typing import Callable, List, Dict, Optional, Union, Any, Generator, AsyncGenerator
from sciagents.agents.agent import Agent
from sciagents.agents.message import AgentInput, AgentOutput, Message, Role
from sciagents.llm import LlmModel, LlmConfig
import httpx # For MCPClient error types
from typing import Callable, List, Dict, Optional, Union, Any, Generator, AsyncGenerator
from sciagents.agents.agent import Agent
from sciagents.agents.message import AgentInput, AgentOutput, Message, Role
from sciagents.llm import LlmModel, LlmConfig
from sciagents.tools.function_tool import FunctionTool
from sciagents.tools.mcp_tool.client import MCPClient
from sciagents.tools.mcp_tool.toolkit import MCPTool
import asyncio

class ChatAgent(Agent):
    """
    A simple chat agent that maintains conversation history and supports streaming.
    Uses LlmModel from llm.py for LLM interactions.
    Can use local FunctionTools and remote MCPTools via an MCPClient.
    """

    def __init__(self,
                 name: str,
                 llm_config: Union[LlmConfig, Dict[str, Any]],
                 tools: Optional[List[Union[Callable, FunctionTool]]] = None,
                 mcp_client: Optional[MCPClient] = None) -> None:
        super().__init__(name, llm_config)
        self.history: List[Dict] = []
        self.tools: List[Union[FunctionTool, MCPTool]] = []
        self.tool_map: Dict[str, Union[FunctionTool, MCPTool]] = {}

        self.mcp_client = mcp_client

        if tools:
            for tool_obj in tools:
                tool_instance: FunctionTool
                if isinstance(tool_obj, FunctionTool):
                    tool_instance = tool_obj
                elif callable(tool_obj):
                    tool_instance = FunctionTool(tool_obj)
                else:
                    print(f"Warning: Local tool object {tool_obj} is not a FunctionTool or callable, skipping.")
                    continue

                tool_name = tool_instance.get_function_name()
                if tool_name in self.tool_map:
                     print(f"Warning: Local tool '{tool_name}' conflicts with an existing tool. It will be ignored.")
                     continue
                self.tools.append(tool_instance)
                self.tool_map[tool_name] = tool_instance

        if self.mcp_client:
            self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        if not self.mcp_client:
            return
        try:
            client_info = f"client for {self.mcp_client.base_url}" if hasattr(self.mcp_client, 'base_url') else "MCP client"
            print(f"ChatAgent '{self.name}': Attempting to list tools from MCP server via {client_info}...")
            remote_tools = self.mcp_client.list_tools()
            for tool_instance in remote_tools:
                tool_name = tool_instance.get_function_name()
                if tool_name in self.tool_map:
                    print(f"Warning: MCPTool '{tool_name}' from server conflicts with an existing tool. It will be ignored.")
                else:
                    self.tools.append(tool_instance)
                    self.tool_map[tool_name] = tool_instance
                    print(f"ChatAgent '{self.name}': Registered MCPTool '{tool_name}'.")
        except Exception as e:
            print(f"ChatAgent '{self.name}': Failed to register MCP tools: {type(e).__name__}: {e}. MCP tools may be unavailable.")

    def run(self, input_data: Union[AgentInput, List[Dict], str], stream: bool = False, *args: Any, **kwargs: Any) -> AgentOutput:
        return self.step(input_data, stream=stream, *args, **kwargs)

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
        def combined_generator() -> Generator[str, None, None]:
            collected_content = []
            final_tool_calls_dict = {}
            for chunk in generator:
                if isinstance(chunk, str):
                    collected_content.append(chunk)
                    yield chunk
                elif isinstance(chunk, list):
                    for tool_call_data in chunk:
                        index = getattr(tool_call_data, 'index', id(tool_call_data))
                        if hasattr(tool_call_data, 'function') and hasattr(tool_call_data.function, 'name') and hasattr(tool_call_data.function, 'arguments'):
                            if index not in final_tool_calls_dict:
                                final_tool_calls_dict[index] = tool_call_data
                            else:
                                final_tool_calls_dict[index].function.arguments += tool_call_data.function.arguments
                        else:
                            print(f"Skipping tool call with unexpected structure: {tool_call_data}")

            final_tool_calls_list = list(final_tool_calls_dict.values())
            if final_tool_calls_list:
                tool_names_to_execute = [getattr(tc.function, 'name', 'unknown') for tc in final_tool_calls_list]
                yield f"[Executing tools: {', '.join(tool_names_to_execute)}]\n"

                self.history.append({"role": "assistant", "tool_calls": final_tool_calls_list})
                tool_call_results_messages = self._execute_tool(final_tool_calls_list)
                self.history.extend(tool_call_results_messages)

                final_generator = self.llm_model.stream_completion(self.history, tools=[t.to_openai_schema() for t in self.tools] if self.tools else None)
                final_content_parts = []
                for final_chunk in final_generator:
                    if isinstance(final_chunk, str):
                        final_content_parts.append(final_chunk)
                        yield final_chunk
                    elif isinstance(final_chunk, list):
                        error_msg = json.dumps({"error": "Nested tool calls in a single assistant turn are not directly supported by this handler."}) + "\n"
                        yield error_msg
                        self.history.append({"role": "assistant", "content": error_msg.strip()})
                        return

                if final_content_parts:
                    self.history.append({"role": "assistant", "content": "".join(final_content_parts)})

            elif collected_content:
                self.history.append({"role": "assistant", "content": "".join(collected_content)})

        return AgentOutput.from_generator(combined_generator())

    async def _handle_async_streaming(self, generator: AsyncGenerator) -> AgentOutput:
        async def combined_generator() -> AsyncGenerator[str, None]:
            collected_content = []
            final_tool_calls_dict = {}
            async for chunk in generator:
                if isinstance(chunk, str):
                    collected_content.append(chunk)
                    yield chunk
                elif isinstance(chunk, list):
                    for tool_call_data in chunk:
                        index = getattr(tool_call_data, 'index', id(tool_call_data))
                        if hasattr(tool_call_data, 'function') and hasattr(tool_call_data.function, 'name') and hasattr(tool_call_data.function, 'arguments'):
                            if index not in final_tool_calls_dict:
                                final_tool_calls_dict[index] = tool_call_data
                            else:
                                final_tool_calls_dict[index].function.arguments += tool_call_data.function.arguments
                        else:
                             print(f"Skipping tool call with unexpected structure: {tool_call_data}")

            final_tool_calls_list = list(final_tool_calls_dict.values())
            if final_tool_calls_list:
                tool_names_to_execute = [getattr(tc.function, 'name', 'unknown') for tc in final_tool_calls_list]
                yield f"[Executing tools: {', '.join(tool_names_to_execute)}]\n"

                self.history.append({"role": "assistant", "tool_calls": final_tool_calls_list})
                tool_call_results_messages = await self._aexecute_tool(final_tool_calls_list)
                self.history.extend(tool_call_results_messages)

                final_generator = self.llm_model.async_stream_completion(self.history, tools=[t.to_openai_schema() for t in self.tools] if self.tools else None)
                final_content_parts = []
                async for final_chunk in final_generator:
                    if isinstance(final_chunk, str):
                        final_content_parts.append(final_chunk)
                        yield final_chunk
                    elif isinstance(final_chunk, list):
                        error_msg = json.dumps({"error": "Nested tool calls in a single assistant turn are not directly supported by this handler."}) + "\n"
                        yield error_msg
                        self.history.append({"role": "assistant", "content": error_msg.strip()})
                        return
                if final_content_parts:
                    self.history.append({"role": "assistant", "content": "".join(final_content_parts)})

            elif collected_content:
                self.history.append({"role": "assistant", "content": "".join(collected_content)})

        return AgentOutput.from_generator(combined_generator())

    def _handle_response(self, response: Dict) -> AgentOutput:
        if response.get("tool_calls"):
            tool_calls_list = response["tool_calls"]
            self.history.append({"role": "assistant", "tool_calls": tool_calls_list})
            tool_call_results = self._execute_tool(tool_calls_list)
            self.history.extend(tool_call_results)

            final_response = self.llm_model.completion(self.history, tools=[t.to_openai_schema() for t in self.tools] if self.tools else None)
            if final_response.get("tool_calls"):
                content = json.dumps({"error": "Multiple levels of tool calls in a single non-streaming step are not supported."})
                self.history.append({"role": "assistant", "content": content})
                return AgentOutput.from_string(content)
            content = final_response.get("content", "")
            self.history.append({"role": "assistant", "content": content})
            return AgentOutput.from_string(content)

        content = response.get("content", "")
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)

    async def _handle_async_response(self, response: Dict) -> AgentOutput:
        if response.get("tool_calls"):
            tool_calls_list = response["tool_calls"]
            self.history.append({"role": "assistant", "tool_calls": tool_calls_list})
            tool_call_results = await self._aexecute_tool(tool_calls_list)
            self.history.extend(tool_call_results)

            final_response = await self.llm_model.async_completion(self.history, tools=[t.to_openai_schema() for t in self.tools] if self.tools else None)
            if final_response.get("tool_calls"):
                content = json.dumps({"error": "Multiple levels of tool calls in a single non-streaming step are not supported."})
                self.history.append({"role": "assistant", "content": content})
                return AgentOutput.from_string(content)
            content = final_response.get("content", "")
            self.history.append({"role": "assistant", "content": content})
            return AgentOutput.from_string(content)

        content = response.get("content", "")
        self.history.append({"role": "assistant", "content": content})
        return AgentOutput.from_string(content)

    def _execute_tool(self, tool_calls: List[Any]) -> List[Dict]:
        results = []
        for tool_call_obj in tool_calls:
            tool_name = "unknown_tool"
            tool_id = getattr(tool_call_obj, 'id', "")
            arguments_str = ""
            result_content = ""
            try:
                if not hasattr(tool_call_obj, 'function') or not hasattr(tool_call_obj.function, 'name'):
                    raise ValueError("Tool call object is missing function name.")
                tool_name = tool_call_obj.function.name

                arguments_str = getattr(tool_call_obj.function, 'arguments', '{}')
                arguments = json.loads(arguments_str) if arguments_str else {}

                tool_instance = self.tool_map.get(tool_name)

                if not tool_instance:
                    result_content = f"Tool {tool_name} not found"
                elif isinstance(tool_instance, FunctionTool):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        exec_output_dict = loop.run_until_complete(asyncio.wait_for(tool_instance.execute(arguments), timeout=10.0))
                        result_content = str(exec_output_dict.get("result", exec_output_dict.get("error", "Execution error")))
                    finally:
                        loop.close()
                elif isinstance(tool_instance, MCPTool):
                    if not self.mcp_client:
                        result_content = f"MCPClient not configured, cannot execute MCPTool {tool_name}"
                    else:
                        mcp_raw_result = self.mcp_client.execute_tool(tool_name, arguments)
                        result_content = str(mcp_raw_result)
                else:
                    result_content = f"Tool {tool_name} is of unknown type: {type(tool_instance)}"

            except json.JSONDecodeError:
                result_content = f"Invalid arguments JSON for tool {tool_name}: {arguments_str}"
            except asyncio.TimeoutError:
                result_content = f"Tool {tool_name} execution timed out"
            except httpx.HTTPStatusError as http_err:
                result_content = f"MCPTool {tool_name} HTTP error: {http_err.response.status_code} - {http_err.response.text}"
            except httpx.RequestError as req_err:
                result_content = f"MCPTool {tool_name} request error: {req_err}"
            except ConnectionError as conn_err:
                 result_content = f"MCPTool {tool_name} connection error: {conn_err}"
            except ValueError as val_err:
                 result_content = f"Tool {tool_name} processing error: {val_err}"
            except RuntimeError as run_err:
                 result_content = f"Tool {tool_name} runtime error: {run_err}"
            except Exception as e:
                result_content = f"Tool {tool_name} execution failed with an unexpected error: {type(e).__name__}: {e}"

            msg = Message(role=Role.TOOL, content=result_content, tool_call_id=tool_id, name=tool_name)
            results.append(msg.to_dict())
        return results

    async def _aexecute_tool(self, tool_calls: List[Any]) -> List[Dict]:
        results = []
        for tool_call_obj in tool_calls:
            tool_name = "unknown_tool"
            tool_id = getattr(tool_call_obj, 'id', "")
            arguments_str = ""
            result_content = ""
            try:
                if not hasattr(tool_call_obj, 'function') or not hasattr(tool_call_obj.function, 'name'):
                    raise ValueError("Tool call object is missing function name.")
                tool_name = tool_call_obj.function.name

                arguments_str = getattr(tool_call_obj.function, 'arguments', '{}')
                arguments = json.loads(arguments_str) if arguments_str else {}

                tool_instance = self.tool_map.get(tool_name)

                if not tool_instance:
                    result_content = f"Tool {tool_name} not found"
                elif isinstance(tool_instance, FunctionTool):
                    exec_output_dict = await asyncio.wait_for(tool_instance.execute(arguments), timeout=10.0)
                    result_content = str(exec_output_dict.get("result", exec_output_dict.get("error", "Execution error")))
                elif isinstance(tool_instance, MCPTool):
                    if not self.mcp_client:
                        result_content = f"MCPClient not configured, cannot execute MCPTool {tool_name}"
                    else:
                        mcp_raw_result = await asyncio.wait_for(self.mcp_client.a_execute_tool(tool_name, arguments), timeout=10.0)
                        result_content = str(mcp_raw_result)
                else:
                    result_content = f"Tool {tool_name} is of unknown type: {type(tool_instance)}"

            except json.JSONDecodeError:
                result_content = f"Invalid arguments JSON for tool {tool_name}: {arguments_str}"
            except asyncio.TimeoutError:
                result_content = f"Tool {tool_name} execution timed out"
            except httpx.HTTPStatusError as http_err:
                result_content = f"MCPTool {tool_name} HTTP error: {http_err.response.status_code} - {http_err.response.text}"
            except httpx.RequestError as req_err:
                result_content = f"MCPTool {tool_name} request error: {req_err}"
            except ConnectionError as conn_err:
                 result_content = f"MCPTool {tool_name} connection error: {conn_err}"
            except ValueError as val_err:
                 result_content = f"Tool {tool_name} processing error: {val_err}"
            except RuntimeError as run_err:
                 result_content = f"Tool {tool_name} runtime error: {run_err}"
            except Exception as e:
                result_content = f"Tool {tool_name} execution failed with an unexpected error: {type(e).__name__}: {e}"

            msg = Message(role=Role.TOOL, content=result_content, tool_call_id=tool_id, name=tool_name)
            results.append(msg.to_dict())
        return results