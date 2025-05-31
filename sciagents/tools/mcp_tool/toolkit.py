from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from ..base_tool import BaseToolkit
from ..function_tool import FunctionTool
from .client import MCPClient

import logging
logger = logging.getLogger("sciagents.mcptoolkit")


class MCPToolkit(BaseToolkit):
    """聚合多个 MCPClient，统一 get_tools。"""

    def __init__(
        self,
        *,
        servers: Optional[List[MCPClient]] = None,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.servers: List[MCPClient] = servers or []
        if config_path:
            self.servers.extend(self._load_servers_from_file(config_path, strict))
        if config_dict:
            self.servers.extend(self._load_servers_from_dict(config_dict, strict))
        self._connected = False

    # ------------------------------------------------------------------
    def _load_servers_from_file(self, path: str, strict: bool) -> List[MCPClient]:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return self._load_servers_from_dict(cfg, strict)

    def _load_servers_from_dict(self, cfg: Dict[str, Any], strict: bool) -> List[MCPClient]:
        res: List[MCPClient] = []
        for name, item in cfg.get("mcp_servers", {}).items():
            if not isinstance(item, dict):
                continue
            cmd_or_url = item.get("url") or item.get("command")
            if not cmd_or_url:
                continue
            res.append(
                MCPClient(
                    command_or_url=cmd_or_url,
                    args=item.get("args", []),
                    env={**os.environ, **item.get("env", {})},
                    timeout=item.get("timeout"),
                    headers=item.get("headers", {}),
                    mode=item.get("mode"),
                    strict=strict,
                )
            )
        return res

    # ------------------------------------------------------------------
    async def connect(self):
        if self._connected:
            return self
        for s in self.servers:
            await s.connect()
        self._connected = True
        return self

    async def disconnect(self):
        if not self._connected:
            return
        for s in self.servers:
            await s.disconnect()
        self._connected = False

    @asynccontextmanager
    async def connection(self):
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    # ------------------------------------------------------------------
    def get_tools(self) -> List[FunctionTool]:
        tools: List[FunctionTool] = []
        for s in self.servers:
            tools.extend(s.get_tools())
        return tools
