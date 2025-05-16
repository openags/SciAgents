import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.agents.chat_agent import ChatAgent
from src.agents.message import AgentInput, Message, Role


# 读取配置文件
with open("config/config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

chat_agent_config = config["agents"]["ChatAgent"]

print("ChatAgent config:", chat_agent_config)


# 构造 AgentInput
messages = [
    Message(role=Role.USER, content="你好，介绍一下你自己吧！")
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
    stream=True
)

print("Agent Name:", agent.name)

# 调用step, stream=True
# 调用step, stream=True
output = agent.step(agent_input)
if hasattr(output.content, "__iter__") and not isinstance(output.content, str):
    # 是生成器，逐步打印
    for chunk in output.content:
        print(chunk, end="", flush=True)
    print()
else:
    # 是字符串，直接打印
    print(output.content)

