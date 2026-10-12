"""A tool set is what the planners (LLM agent and fallback) call.

* :class:`LocalToolSet` calls the :class:`~peakwhale_helm.toolbox.Toolbox` in this process.
* :class:`MCPToolSet` starts ``helm-mcp`` as a child process and calls it over MCP,
  exactly like Claude Desktop or any other MCP client would.

Both raise an exception when a tool fails. The message explains the problem.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from .toolbox import Toolbox


class ToolSet(Protocol):
    @property
    def names(self) -> list[str]: ...

    async def call(self, name: str, args: dict[str, Any]) -> str: ...

    def langchain_tools(self) -> list[Any]: ...


def content_to_text(content: Any) -> str:
    """Turn LangChain / MCP tool output (str, dict or list of blocks) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, tuple) and len(content) == 2:  # (content, artifact)
        return content_to_text(content[0])
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(t for t in (content_to_text(item) for item in content) if t).strip()
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


class LocalToolSet:
    def __init__(self, toolbox: Toolbox):
        self.toolbox = toolbox

    @property
    def names(self) -> list[str]:
        return self.toolbox.tool_names

    async def call(self, name: str, args: dict[str, Any]) -> str:
        return await asyncio.to_thread(self.toolbox.call, name, dict(args or {}))

    def langchain_tools(self) -> list[Any]:
        from langchain_core.tools import StructuredTool

        tools = []
        for name in self.names:
            tools.append(
                StructuredTool.from_function(
                    func=self.toolbox.function(name),
                    name=name,
                    description=self.toolbox.description(name),
                )
            )
        return tools

    async def aclose(self) -> None:
        self.toolbox.close()


class MCPToolSet:
    """Tools loaded from the Helm MCP server running as a child process.

    One background task owns the MCP session from start to end. This matters
    because anyio requires a session to be closed by the task that opened it,
    and web apps (like Chainlit) often start and stop a chat in different tasks.
    """

    def __init__(self, tools: list[Any], owner: asyncio.Task | None = None, stop: asyncio.Event | None = None):
        self._tools = {t.name: t for t in tools}
        self._owner = owner
        self._stop = stop

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def langchain_tools(self) -> list[Any]:
        return list(self._tools.values())

    async def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool {name!r}. Available tools: {', '.join(self._tools)}")
        return content_to_text(await tool.ainvoke(dict(args or {})))

    @classmethod
    async def start(cls, config_path: str | None = None, cwd: str | Path | None = None) -> "MCPToolSet":
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        args = ["-m", "peakwhale_helm.mcp_server"]
        if config_path:
            args += ["--config", str(config_path)]
        connection = {
            "transport": "stdio",
            "command": sys.executable,
            "args": args,
            "cwd": str(cwd or Path.cwd()),
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        client = MultiServerMCPClient({"helm": connection}, tool_name_prefix=False)
        ready, stop = asyncio.Event(), asyncio.Event()
        result: dict[str, Any] = {}

        async def own_session() -> None:
            try:
                async with client.session("helm") as session:
                    result["tools"] = await load_mcp_tools(session, server_name="helm", tool_name_prefix=False)
                    ready.set()
                    await stop.wait()
            except Exception as exc:
                result["error"] = exc
            finally:
                ready.set()

        owner = asyncio.create_task(own_session(), name="helm-mcp-session")
        await ready.wait()
        if "tools" not in result:
            await asyncio.wait({owner}, timeout=5)
            raise RuntimeError(f"Could not start the Helm MCP server: {result.get('error')}")
        return cls(result["tools"], owner, stop)

    async def aclose(self) -> None:
        if self._owner is None or self._stop is None:
            return
        owner, self._owner = self._owner, None
        self._stop.set()
        done, _ = await asyncio.wait({owner}, timeout=5)
        if not done:
            owner.cancel()
