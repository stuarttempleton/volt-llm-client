# volt-llm-client

**volt-llm-client** is a lightweight Python client for interacting with local LLM APIs such as [Ollama](https://ollama.com/) or [OpenWebUI](https://github.com/open-webui/open-webui).  
It supports prompt completion, multi-turn conversations, and model listing — perfect for scripts, prototypes, or CLI tools.

---

## Features

- Send single prompts or full conversations
- Compatible with Ollama/OpenWebUI-style APIs
- Optional MCP tool calling via a **Docker Desktop MCP Toolkit** gateway
- Token-based authentication support
- Clean log output using [volt-logger](https://github.com/stuarttempleton/volt-logger)
- Minimal dependencies (`requests` only; `fastmcp` for the optional `mcp` extra)

---

## Installation

Install both `volt-llm-client` and its logging dependency:

```bash
pip install volt-llm-client
````

For local development:

```bash
git clone https://github.com/stuarttempleton/volt-llm-client.git
cd volt-llm-client
pip install -r requirements.txt
```

---

## Usage Example

```python
import os
from voltllmclient import LLMClient

llm = LLMClient(
    token=os.getenv("LLM_API_TOKEN"),
    model="Gemma4"
)

reply = llm.send_prompt("What is the capital of France?")
print(reply)
```

You can also send a full conversation:

```python
messages = [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Who won the World Cup in 2018?" }
]

response = llm.send_conversation(messages)
print(response)
```

---

## Maintaining Context with LLMConversation

```python
from voltllmclient import LLMConversation
import os

conv = LLMConversation(model="gemma4", token=os.getenv("LLM_API_TOKEN"))

response = conv.send("What is quantum computing?")
print(response)

response = conv.send_with_full_context("How is it different from classical computing?")
print(response)

conv.save_transcript("session.json")
```

---

## MCP Tool Calling

Lets the model call tools from your MCP servers. Requires the `mcp` extra:

```bash
pip install volt-llm-client[mcp]
```

```python
from voltllmclient import LLMClient, MCPToolProvider

# Talks to a Docker Desktop MCP Toolkit gateway. Pass the name of the profile holding
# the servers you want — without one, the gateway only exposes its own meta-tools.
provider = MCPToolProvider(profile="my_profile")

if provider.connect():
    llm = LLMClient(model="gemma4:26b", mcp=provider)
    print(llm.send_with_tools([{"role": "user", "content": "What is in my log files?"}]))

provider.close()
```

List your profiles with `docker mcp profile list`, and check what a profile exposes with
`docker mcp tools ls --gateway-arg=--profile=<name>`.

A profile is required. Without one the gateway advertises only its own meta-tools (`mcp-add`,
`mcp-config-set`, ...) which can rewrite your MCP setup, so `connect()` refuses and returns `False`
rather than handing those to a model. Pass `args=` instead to drive a non-Docker stdio MCP server:

```python
MCPToolProvider(command="my-mcp-server", args=["serve", "--stdio"])
```

Config is per instance — nothing is read from the environment — so several providers with different
profiles and filters can run side by side in one process.

`send_with_tools` loops until the model stops requesting tools (capped by `max_tool_rounds=5`).
If the gateway is unreachable, `connect()` logs a warning and returns `False` — pass `mcp=None` and
the prompt is still answered without tools.

The gateway's own stderr is captured to a log file rather than the console (it is noisy on success
and prints a stack trace when Docker Desktop is down). On failure the warning quotes the relevant
line from it and tells you where the full output is.

Each provider gets its own temp log (`<tempdir>/volt-mcp-gateway-<pid>-<random>.log`) so concurrent
instances never clobber each other. It is deleted on `close()` unless the connection failed, in
which case it is kept for you to read. Pass `MCPToolProvider(log_file="path/to/gateway.log")` to
choose the location yourself — a log you supply is never deleted.

### Narrowing the tool list

Every advertised tool costs prompt tokens on *each* round, and a gateway with dozens of
similarly-named tools makes a small model more likely to pick the wrong one. Filter client-side:

```python
MCPToolProvider(include="get_*,search_*")        # glob, comma-separated
MCPToolProvider(tools=["get_file", "list_dirs"]) # exact names
```

### CLI

```bash
python -m voltllmclient.client gemma4 "What is the capital of France?"

# with tools from an MCP profile, narrowed to the ones the model needs
python -m voltllmclient.client gemma4 "What is in my log files?" \
    --profile my_profile --tools "get_*,search_*"
```

| Option | Default | Meaning |
|---|---|---|
| `--url` | `http://localhost:11434` | API base url (domain:port, not an endpoint) |
| `--profile` | none | Docker MCP profile to expose; omit and no MCP is used at all |
| `--tools` | all | tool filter glob, comma separated |
| `--timeout` | `120` | request timeout in seconds; raise it if prompt processing is slow |

---

## Environment Variables

Set your API token via environment variable:

**Unix/macOS:**

```bash
export LLM_API_TOKEN=your_token_here
```

**PowerShell:**

```powershell
$env:LLM_API_TOKEN = "your_token_here"
```

---

## License

[MIT](LICENSE)

