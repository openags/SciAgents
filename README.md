# SciAgents

SciAgents is an extensible multi-agent framework designed for scientific research scenarios. It leverages large language models (LLMs) to automate research tasks and can be integrated with robotic systems for advanced scientific workflows.

## Features

- Supports various LLM providers (OpenAI, Azure, Gemini, etc.)
- Modular agent and tool system for easy extension
- Flexible configuration for different research needs
- Ready-to-use test scripts and Jupyter notebooks
- Can be combined with robots for automated scientific experiments

## Installation

```bash
pip install sciagents
```

## Quick Start

Below is a minimal example of using `ChatAgent` in your project.  
**Note:** Make sure you have a valid `config/config.yml` with your LLM API keys and model info.

```python
import os
from sciagents.agents.chat_agent import ChatAgent
from sciagents.agents.message import AgentInput, Message, Role
import yaml

# Load config
config_path = os.path.join("config", "config.yml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
chat_agent_config = config["agents"]["ChatAgent"]

# Build agent input
messages = [Message(role=Role.USER, content="Introduce yourself, please.")]
agent_input = AgentInput(messages=messages)

# Create ChatAgent instance
agent = ChatAgent(
    name="DemoChatAgent",
    llm_config={
        "model": chat_agent_config["model"],
        "api_key": chat_agent_config["api_key"],
        "api_base": chat_agent_config["url"],
        **chat_agent_config.get("model_config_dict", {})
    },
    stream=True
)

# Get response
output = agent.step(agent_input)
if hasattr(output.content, "__iter__") and not isinstance(output.content, str):
    for chunk in output.content:
        print(chunk, end="", flush=True)
    print()
else:
    print(output.content)
```

## Project Structure

- `sciagents/`: Core code (agents, tools, LLM interfaces)
- `config/`: Configuration files
- `test/`: Test scripts and examples

## License

This project is licensed under the MIT License.