# conversation.py
# 05/17/2025 - Voltur
#
# Maintains multi-turn conversation context with a local LLM via OpenWebUI/Ollama.
# Depends on LLMClient.py


from .client import LLMClient
import sys

class LLMConversation:
    def __init__(self, model="Gemma4", system_prompt=None, token=None, base_url="http://localhost:11434", mcp=None):
        # mcp takes either a Docker MCP profile name, which we build and own a provider for, or a
        # provider the caller built and manages themselves.
        self._own_mcp = None
        if isinstance(mcp, str):
            from .mcptools import MCPToolProvider
            self._own_mcp = MCPToolProvider(profile=mcp)
            # Tools are best effort: a dead gateway leaves us a working chat.
            mcp = self._own_mcp if self._own_mcp.connect() else None
        self.client = LLMClient(model=model, token=token, base_url=base_url, mcp=mcp)
        self.use_tools = mcp is not None
        self.messages = [
            {
                "role": "system",
                "content": system_prompt or (
                    "You are a helpful and knowledgeable AI assistant. "
                    "Provide clear, concise, and accurate responses. "
                    "When appropriate, ask clarifying questions or provide examples."
                )
            }
        ]

    def send(self, user_content, send_everything=False, assistant_only=False):
        if assistant_only:
            # Use only system + previous assistant messages
            prompt = [self.messages[0]]  # system prompt
            prompt += [m for m in self.messages if m["role"] == "assistant"]
            prompt.append({"role": "user", "content": user_content})

        elif send_everything:
            prompt = self.messages + [{"role": "user", "content": user_content}]

        else:
            prompt = [
                {"role": "system", "content": self.messages[0]["content"]},
                {"role": "user", "content": user_content}
            ]

        if self.use_tools:
            reply = self.client.send_with_tools(prompt)
        else:
            reply = self.client.send_conversation(prompt)

        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": reply})

        return reply


    def close(self):
        # Only shut down a provider we built; a caller-supplied one is theirs to close.
        if self._own_mcp:
            self._own_mcp.close()
            self._own_mcp = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send_with_full_context(self, user_content):
        return self.send(user_content, send_everything=True)
    
    def send_with_summary_context(self, user_content):
        return self.send(user_content, assistant_only=True)

    def save_transcript(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2)

    def load_transcript(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)