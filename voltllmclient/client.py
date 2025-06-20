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
    def __init__(self, api_url="http://localhost:3000/api/chat/completions",token=None, model="Gemma3", temperature=0.2):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature or 0.2
        self.bearer_token = token or os.getenv("LLM_API_TOKEN", "")
    
    def get_models(self):
        # this returns the response. of openwebui you'll want to use something like:
        # for model in result['data']: Logger.log(f"{model.get('name', '??')} - {model.get('id', '??')}")
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.get("http://localhost:3000/api/models", headers=headers)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {response.text}")

        return None
    
    def send_prompt(self, prompt, system_prompt=None):
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        data = {
        "model": self.model,
        "messages": [
            { "role": "system", "content": system_prompt or "You are a senior application security engineer." },
            { "role": "user", "content": prompt }
        ],
        "temperature": self.temperature, 
        "stream": False
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            Logger.error(f"Request failed: {e}")
        except KeyError:
            Logger.error(f"Unexpected response: {response.text}")

        return None
    
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

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]



if __name__ == "__main__":
    if len(sys.argv) != 3:
        Logger.help(f"Usage: python {sys.argv[0]} <model> <prompt>")
        sys.exit(1)
    
    token = os.getenv("LLM_API_TOKEN")
    if not token:
        # Need a hand?
        # Powershell user permanent:
        # [System.Environment]::SetEnvironmentVariable("LLM_API_TOKEN", "your-secret-token-here", "User")
        # Powershell user temp:
        # $env:LLM_API_TOKEN = "your-secret-token-here"
        Logger.warn("Missing API token. Please set LLM_API_TOKEN as a user environment variable.")
        sys.exit(1)
    
    llm = LLMClient(token=os.getenv("LLM_API_TOKEN"), model=sys.argv[1])
    reply = llm.send_prompt(sys.argv[2])

    if reply:
        Logger.log(f"🤖 {reply}")
