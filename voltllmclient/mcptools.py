# mcptools.py
# 07/25/2026 - Voltur
#
# Exposes MCP tools to an LLM client. Talks to a Docker Desktop MCP Toolkit
# gateway (or any stdio MCP server) and presents tools in OpenAI function form.
# Optional: requires the 'mcp' extra (pip install volt-llm-client[mcp]).


import asyncio
import fnmatch
import os
import tempfile
import threading
from voltlogger import Logger

# The Docker MCP gateway panics without these on Windows; stdio transports pass no env by default.
GATEWAY_ENV_KEYS = ("LOCALAPPDATA", "APPDATA", "ProgramData", "ProgramFiles", "ProgramFiles(x86)",
                    "USERPROFILE", "PATH", "HOME", "SystemRoot", "TEMP")


class MCPToolProvider:
    def __init__(self, profile=None, command="docker", args=None, tools=None,
                 include=None, connect_timeout=120, timeout=60, env=None, log_file=None):
        # Without a profile the Docker gateway serves only its own meta-tools (mcp-add,
        # mcp-config-set, ...) which let a model rewrite the user's MCP setup. So a profile
        # is required, unless args= points at some other stdio MCP server entirely.
        self.profile = profile
        self.args = args or ["mcp", "gateway", "run"] + (["--profile", profile] if profile else [])
        self.command = command
        self._configured = bool(profile or args)
        self.env = env or {k: os.environ[k] for k in GATEWAY_ENV_KEYS if k in os.environ}
        self.tools = tools
        self.include = include
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        # The gateway is chatty (and panics loudly) on stderr; keep it out of the console but
        # hold on to it so connect() can report why it failed. Pass log_file to redirect it.
        # mkstemp keeps concurrent providers (threads or processes) off each other's log.
        self._own_log = log_file is None
        if self._own_log:
            fd, self._log_path = tempfile.mkstemp(prefix=f"volt-mcp-gateway-{os.getpid()}-", suffix=".log")
            os.close(fd)
        else:
            self._log_path = log_file
        self._client = None
        self._specs = None
        self._failed = False
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def _run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout or self.timeout)

    def connect(self):
        if not self._configured:
            Logger.warn("MCP not configured: pass profile= (or args=) to expose tools")
            return False
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StdioTransport
        except ImportError:
            Logger.warn("MCP unavailable: fastmcp not installed (pip install volt-llm-client[mcp])")
            return False
        try:
            from pathlib import Path
            # fastmcp appends, so clear it first or we report a previous run's failure.
            try:
                open(self._log_path, "w").close()
            except OSError:
                pass
            client = Client(StdioTransport(self.command, self.args, env=self.env,
                                           log_file=Path(self._log_path)))
            self._run(client.__aenter__(), self.connect_timeout)
            self._client = client
            Logger.log(f"MCP connected: {len(self.tool_specs())} tools")
            return True
        except Exception as e:
            reason = self._gateway_error() or e
            Logger.warn(f"MCP unavailable: {reason}")
            Logger.warn(f"Gateway output: {self._log_path}")
            self._client = None
            self._failed = True  # keep the log around for the user to read
            return False

    def _gateway_error(self):
        # Pull the most useful line out of the gateway's stderr (e.g. "Docker Desktop is not running").
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
        except OSError:
            return None
        # A Go panic states the cause up front; everything after it is stack trace.
        for line in lines:
            if line.startswith("panic:"):
                return line.split(" [recovered")[0]
        # Otherwise take the last line that is not progress noise or a stack frame.
        for line in reversed(lines[-40:]):
            if line.startswith(("-", ">", "goroutine", "github.com/", "sync/", "runtime/", "main.")):
                continue
            if "0x" in line or line.endswith("()"):
                continue
            return line
        return None

    def _keep(self, name):
        if self.tools:
            return name in self.tools
        if self.include:
            return any(fnmatch.fnmatch(name, p.strip()) for p in self.include.split(","))
        return True

    def tool_specs(self):
        if self._specs is not None:
            return self._specs
        if not self._client:
            return []
        try:
            tools = self._run(self._client.list_tools())
        except Exception as e:
            Logger.warn(f"MCP tool listing failed: {e}")
            return []
        self._specs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}}
                }
            }
            for t in tools if self._keep(t.name)
        ]
        if self.tools or self.include:
            Logger.log(f"MCP tool filter: {len(self._specs)}/{len(tools)} tools advertised")
        return self._specs

    def call(self, name, arguments):
        if not self._client:
            return "Error: MCP is not connected."
        try:
            result = self._run(self._client.call_tool(name, arguments or {}))
        except Exception as e:
            # Hand the failure to the model as text so it can recover or explain.
            Logger.warn(f"MCP tool '{name}' failed: {e}")
            return f"Error: {e}"
        texts = [b.text for b in getattr(result, "content", []) or [] if getattr(b, "text", None)]
        return "\n".join(texts) if texts else str(getattr(result, "data", ""))

    def close(self):
        if self._client:
            try:
                self._run(self._client.__aexit__(None, None, None), self.timeout)
            except Exception as e:
                Logger.warn(f"MCP shutdown failed: {e}")
            self._client = None
        # Only tidy up our own temp log, and only when it holds nothing worth reading.
        if self._own_log and not self._failed:
            try:
                os.unlink(self._log_path)
            except OSError:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
