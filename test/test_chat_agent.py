import sys
import os
import yaml
from pathlib import Path

# Add project root to Python path
notebook_dir = Path().absolute()  # Gets the current directory
project_root = str(notebook_dir.parent)  # Go up one level to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Get config path
config_path = os.path.join(project_root, "config", "config.yml")
print("Project root:", project_root)
print("Config path:", config_path)

from sciagents.agents.chat_agent import ChatAgent
from sciagents.agents.message import AgentInput, Message, Role
from sciagents.tools import function_tool


@function_tool(name_override="fetch_weather")
async def fetch_weather(location: str) -> str:
    return f"Sunny in {location}"

def get_stock_price(symbol: str) -> float:
    return 100.2


# 读取配置文件
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

chat_agent_config = config["agents"]["ChatAgent"]

print("ChatAgent config:", chat_agent_config)


# 构造 AgentInput
messages = [
    Message(role=Role.USER, content="你好，获取苹果股票价格，和上海的天气！")
]
agent_input = AgentInput(messages=messages)

# 创建 ChatAgent 实例
agent = ChatAgent(
    name="TestChatAgent",
    llm_config={
        "model": chat_agent_config["model"],
        "api_key": chat_agent_config["api_key"],
        "api_base": chat_agent_config["url"],
        **chat_agent_config.get("model_config_dict", {})
    },
    tools=[fetch_weather, get_stock_price]
)



# test a_step, stream=True, 异步方法
async def test_step():
    output = await agent.a_step(agent_input, stream=True)
    if hasattr(output.content, "__aiter__"):  # 检查是否为异步生成器
        async for chunk in output.content:
            print(chunk, end="", flush=True)
        print()
    elif hasattr(output.content, "__iter__") and not isinstance(output.content, str):
        for chunk in output.content:
            print(chunk, end="", flush=True)
        print()
    else:
        print(output.content)

import asyncio

if __name__ == "__main__":
    asyncio.run(test_step())