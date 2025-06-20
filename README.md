# volt-llm-client

**volt-llm-client** is a lightweight Python client for interacting with local LLM APIs such as [Ollama](https://ollama.com/) or [OpenWebUI](https://github.com/open-webui/open-webui).  
It supports prompt completion, multi-turn conversations, and model listing — perfect for scripts, prototypes, or CLI tools.

---

## 🔧 Features

- Send single prompts or full conversations
- Compatible with Ollama/OpenWebUI-style APIs
- Token-based authentication support
- Clean log output using [volt-logger](https://github.com/stuarttempleton/volt-logger)
- Minimal dependencies (`requests` only)

---

## 📦 Installation

Install directly from GitHub (also pulls in `volt-logger`):

```bash
pip install git+https://github.com/stuarttempleton/volt-llm-client.git
````

For local development:

```bash
git clone https://github.com/stuarttempleton/volt-llm-client.git
cd volt-llm-client
pip install -r requirements.txt
```

---

## 🧪 Usage Example

```python
import os
from voltllmclient import LLMClient

llm = LLMClient(
    token=os.getenv("LLM_API_TOKEN"),
    model="Gemma3"
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

## 🗣️ Maintaining Context with LLMConversation

```python
from voltllmclient import LLMConversation
import os

conv = LLMConversation(model="gemma3", token=os.getenv("LLM_API_TOKEN"))

response = conv.send("What is quantum computing?")
print(response)

response = conv.send_with_full_context("How is it different from classical computing?")
print(response)

conv.save_transcript("session.json")
```

---

## 🔐 Environment Variables

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

## 🪪 License

[MIT](LICENSE)

