# client.py
# 05/17/2025 - Voltur
#
# An interface to local (ollama style) LLM API.
# 


import requests
import sys
import os
from voltlogger import Logger

class LLMClient:
    def __init__(self, base_url="http://localhost:3000", token=None, model="Gemma3", temperature=0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature or 0.2
        self.bearer_token = token or os.getenv("LLM_API_TOKEN", "")
        self.api_type = None
        self.endpoints = {}
        self._detect_api_type()
    
    def get_models(self):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.get(self.endpoints['models'], headers=headers)
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
        data = {
            "model": self.model,
            "messages": [
                { "role": "system", "content": system_prompt or "You are a helpful and friendly AI assistant." },
                { "role": "user", "content": prompt }
            ],
            "temperature": self.temperature,
            "stream": False
        }
        try:
            Logger.error(f"API Type: {self.api_type}, Endpoint: {self.endpoints['chat']}")
            response = requests.post(self.endpoints['chat'], headers=headers, json=data)
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

    def send_conversation(self, messages):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        response = requests.post(self.endpoints['chat'], headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return self.extract_content(result)

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
    if len(sys.argv) != 3:
        Logger.help(f"Usage: python {sys.argv[0]} <model> <prompt>")
        sys.exit(1)

    token = os.getenv("LLM_API_TOKEN")
    if not token:
        Logger.warn("Missing API token. Please set LLM_API_TOKEN as a user environment variable.")
        sys.exit(1)

    # Only pass base_url (domain:port), not full endpoint
    llm = LLMClient(base_url="http://localhost:3000", token=token, model=sys.argv[1])
    reply = llm.send_prompt(sys.argv[2])

    if reply:
        Logger.log(f"🤖 {reply}")
