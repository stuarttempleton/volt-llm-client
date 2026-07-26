# conversation.py
# 05/17/2025 - Voltur
#
# Maintains multi-turn conversation context with a local LLM via OpenWebUI/Ollama.
# Depends on LLMClient.py


from .client import LLMClient
import sys

class LLMConversation:
    def __init__(self, model="Gemma4", system_prompt=None, token=None, base_url="http://localhost:11434", mcp=None,
                 mcp_include=None, max_tool_result_chars=None):
        # mcp takes either a Docker MCP profile name, which we build and own a provider for, or a
        # provider the caller built and manages themselves.
        # Retained tool results are kept whole by default. They can be large (a VRChat group
        # listing runs past 100KB) and send_everything=True resends history every turn, so on a
        # small-context model set max_tool_result_chars to trim what persists.
        self.max_tool_result_chars = max_tool_result_chars
        self._own_mcp = None
        if isinstance(mcp, str):
            from .mcptools import MCPToolProvider
            # mcp_include narrows which tools are advertised. Worth setting: the specs go out
            # on every request, so a broad profile is a fixed cost on each one.
            self._own_mcp = MCPToolProvider(profile=mcp, include=mcp_include)
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
            # Use only system + previous assistant messages. Skip tool-call turns:
            # their results are role="tool" and would be filtered out here, and an
            # orphaned tool_calls message is rejected by OpenAI-compatible endpoints.
            prompt = [self.messages[0]]  # system prompt
            prompt += [m for m in self.messages
                       if m["role"] == "assistant" and not m.get("tool_calls")]
            prompt.append({"role": "user", "content": user_content})

        elif send_everything:
            prompt = self.messages + [{"role": "user", "content": user_content}]

        else:
            prompt = [
                {"role": "system", "content": self.messages[0]["content"]},
                {"role": "user", "content": user_content}
            ]

        tool_transcript = []
        if self.use_tools:
            reply = self.client.send_with_tools(prompt, transcript=tool_transcript)
        else:
            reply = self.client.send_conversation(prompt)

        self.messages.append({"role": "user", "content": user_content})
        # Keep the tool round-trip between the user turn and the answer. Dropping it
        # leaves the model with no evidence it ever called a tool, so on the next turn
        # it re-runs the same lookups or insists it has no access to the data.
        self.messages.extend(self._trim_tool_result(m) for m in tool_transcript)
        self.messages.append({"role": "assistant", "content": reply})

        return reply

    def _trim_tool_result(self, message):
        # The model already summarised the full result in its reply this turn; later turns
        # only need enough to know the call happened and roughly what came back.
        limit = self.max_tool_result_chars
        if not limit or message.get("role") != "tool":
            return message
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= limit:
            return message
        dropped = len(content) - limit
        return {**message,
                "content": f"{content[:limit]}\n[... {dropped} chars truncated from tool result ...]"}


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