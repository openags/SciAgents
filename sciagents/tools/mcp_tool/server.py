from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, List, Optional

from mcp.server.fastmcp import FastMCP

from ..base_tool import BaseToolkit


class MCPServer:
    """装饰器：把类/Toolkit 的一组方法注册进 FastMCP Server。
    
    将一个普通的工具类转换为 MCP 服务器。可以通过两种方式使用：
    1. 指定 function_names 参数，明确列出要注册的方法名称
    2. 对于继承了 BaseToolkit 的类，会自动注册 get_tools() 返回的所有工具
    
    装饰后的类实例将拥有 mcp 属性，可以通过 inst.mcp.run() 启动服务器。
    """

    def __init__(
        self,
        function_names: Optional[List[str]] = None,
        server_name: Optional[str] = None,
    ) -> None:
        self.function_names = function_names
        self.server_name = server_name

    # ------------------------------------------------------------------
    def _wrap(self, fn: Callable[..., Any]):
        """保持签名 + 支持 async 的薄包装，供 FastMCP introspect。"""
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)
        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    # ------------------------------------------------------------------
    def __call__(self, cls):
        """Decorate class: init 时注入 FastMCP 并注册方法。"""

        orig_init = cls.__init__

        def new_init(inst, *args, **kwargs):
            orig_init(inst, *args, **kwargs)
            inst.mcp = FastMCP(self.server_name or cls.__name__)

            fn_names = self.function_names
            if fn_names is None:
                if isinstance(inst, BaseToolkit):
                    fn_names = [t.get_function_name() for t in inst.get_tools()]
                else:
                    raise ValueError("function_names 为空且类未继承 BaseToolkit")

            for name in fn_names:
                fn = getattr(inst, name, None)
                if fn is None or not callable(fn):
                    raise ValueError(f"{name} 不存在或不可调用")
                inst.mcp.tool(name=name)(self._wrap(fn))
            
            # 添加 run_mcp_server 方法，与 BaseToolkit 接口保持一致
            def run_mcp_server(self_inst, mode="stdio"):
                """启动 MCP 服务器。
                
                Args:
                    mode: 服务器运行模式，可以是 "stdio", "sse", 或 "streamable-http"
                """
                self_inst.mcp.run(mode=mode)
            
            # 为实例添加运行方法
            inst.run_mcp_server = run_mcp_server.__get__(inst)

        cls.__init__ = new_init
        return cls
