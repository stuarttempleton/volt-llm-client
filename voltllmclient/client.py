# client.py
# 05/17/2025 - Voltur
#
# An interface to local (ollama style) LLM API.
# 


import requests
import sys
import os
import json
from voltlogger import Logger

class LLMClient:
    def __init__(self, base_url="http://localhost:11434", token=None, model="Gemma4", temperature=0.2, timeout=120, mcp=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature or 0.2
        self.bearer_token = token or os.getenv("LLM_API_TOKEN", "")
        self.timeout = timeout
        self.mcp = mcp
        self.api_type = None
        self.endpoints = {}
        self._detect_api_type()

    def _build_payload(self, messages, tools=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        if self.api_type == "ollama":
            # Ollama takes generation settings under "options", not top-level.
            payload["options"] = {"temperature": self.temperature}
        else:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
        return payload


    def get_models(self):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.get(self.endpoints['models'], headers=headers, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {getattr(response, 'text', '')}")
        return None
    
    def send_prompt(self, prompt, system_prompt=None):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        data = self._build_payload([
            { "role": "system", "content": system_prompt or "You are a helpful and friendly AI assistant." },
            { "role": "user", "content": prompt }
        ])
        try:
            #Logger.error(f"API Type: {self.api_type}, Endpoint: {self.endpoints['chat']}")
            response = requests.post(self.endpoints['chat'], headers=headers, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return self.extract_content(result)
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {getattr(response, 'text', '')}")
        return None
    
    def extract_content(self, result):
        if self.api_type == "openwebui":
            return result["choices"][0]["message"]["content"]
        elif self.api_type == "ollama":
            return result["message"]["content"]
        else:
            # fallback: try both
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            return result.get("message", {}).get("content", "")

    def extract_message(self, result):
        # Like extract_content, but returns the whole assistant message (tool_calls included).
        if self.api_type == "openwebui":
            return result["choices"][0]["message"]
        elif self.api_type == "ollama":
            return result["message"]
        else:
            # fallback: try both
            if "choices" in result:
                return result["choices"][0]["message"]
            return result.get("message", {})

    def _normalize_tool_calls(self, message):
        # Ollama sends arguments as a dict; Open WebUI sends a JSON string and supplies an id.
        calls = []
        for call in (message or {}).get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    Logger.warn(f"Could not parse tool arguments: {args}")
                    args = {}
            calls.append((call.get("id"), fn.get("name"), args or {}))
        return calls

    def send_with_tools(self, messages, max_tool_rounds=5):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        tools = self.mcp.tool_specs() if self.mcp else []
        messages = list(messages)
        try:
            for _ in range(max_tool_rounds + 1):
                payload = self._build_payload(messages, tools=tools)
                response = requests.post(self.endpoints['chat'], headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                message = self.extract_message(response.json())
                calls = self._normalize_tool_calls(message)
                if not calls:
                    return message.get("content", "")
                messages.append(message)
                for call_id, name, args in calls:
                    Logger.log(f"MCP tool call: {name}({args})")
                    reply = {"role": "tool", "content": self.mcp.call(name, args)}
                    if call_id:
                        reply["tool_call_id"] = call_id
                    if self.api_type == "ollama":
                        reply["name"] = name
                    messages.append(reply)
            Logger.warn(f"Stopped after {max_tool_rounds} tool rounds.")
            return message.get("content", "")
        except requests.Timeout as e:
            Logger.error(f"Request timed out after {self.timeout}s with {len(tools)} tools advertised: {e}")
            if len(tools) > 20:
                Logger.warn("Narrow the tool list (include=/tools=) or raise timeout.")
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {getattr(response, 'text', '')}")
        return None

    def send_conversation(self, messages):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        payload = self._build_payload(messages)
        try:
            response = requests.post(self.endpoints['chat'], headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return self.extract_content(result)
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {getattr(response, 'text', '')}")
        return None

    def _detect_api_type(self):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        # Try Open WebUI first
        try:
            resp = requests.get(f"{self.base_url}/api/models", headers=headers, timeout=2)
            if resp.status_code == 200 and (isinstance(resp.json(), dict) and ("data" in resp.json() or "choices" in resp.json())):
                self.api_type = "openwebui"
                self.endpoints = {
                    'models': f"{self.base_url}/api/models",
                    'chat': f"{self.base_url}/api/chat/completions"
                }
                return
        except Exception:
            pass
        # Try Ollama
        try:
            resp = requests.get(f"{self.base_url}/api/tags", headers=headers, timeout=2)
            if resp.status_code == 200:
                self.api_type = "ollama"
                self.endpoints = {
                    'models': f"{self.base_url}/api/tags",
                    'chat': f"{self.base_url}/api/chat"
                }
                return
        except Exception:
            pass
        # Unknown API
        self.api_type = "unknown"
        self.endpoints = {
            'models': f"{self.base_url}/api/models",
            'chat': f"{self.base_url}/api/chat/completions"
        }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send a prompt to a local LLM, optionally with MCP tools.")
    parser.add_argument("--model", default="Gemma4", help="model name, e.g. gemma4")
    parser.add_argument("prompt", help="the prompt to send")
    parser.add_argument("--url", default="http://localhost:11434", help="API base url (domain:port, not an endpoint)")
    parser.add_argument("--profile", help="Docker MCP profile holding the servers to expose; omit for no tools")
    parser.add_argument("--tools", help="tool filter glob, comma separated, e.g. 'get_*,search_*'")
    # Enough for a cold model load plus a slow tool call; raise it for big prompts on slow hardware.
    parser.add_argument("--timeout", type=int, default=120, help="request timeout in seconds (default: 120)")
    args = parser.parse_args()

    # Model replies routinely contain emoji; Windows consoles default to cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    # No profile means no gateway to talk to, so skip MCP entirely.
    provider = None
    if args.profile:
        try:
            from .mcptools import MCPToolProvider
        except ImportError:
            from mcptools import MCPToolProvider  # running this file directly
        provider = MCPToolProvider(profile=args.profile, include=args.tools)
    try:
        # MCP is best effort: if the gateway is down we still answer the prompt.
        mcp = provider if provider and provider.connect() else None
        llm = LLMClient(base_url=args.url, model=args.model, mcp=mcp, timeout=args.timeout)
        reply = llm.send_with_tools([
            {"role": "system", "content": "You are a helpful and friendly AI assistant."},
            {"role": "user", "content": args.prompt}
        ])

        if reply:
            Logger.log(f"🤖 {reply}")
    finally:
        if provider:
            provider.close()
