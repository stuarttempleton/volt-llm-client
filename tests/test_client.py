import pytest
import requests
from unittest.mock import patch
from voltllmclient.client import LLMClient

# Sample response data
mock_model_response = {
    "data": [{"id": "gemma:7b", "name": "Gemma3"}]
}

mock_chat_response = {
    "choices": [
        {"message": {"content": "Hello from the LLM!"}}
    ]
}


def test_get_models_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_model_response

        client = LLMClient(token="fake-token")
        result = client.get_models()

        assert result == mock_model_response
        mock_get.assert_called_once()


def test_send_prompt_success():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_chat_response

        client = LLMClient(token="fake-token")
        result = client.send_prompt("Hello")

        assert result == "Hello from the LLM!"
        mock_post.assert_called_once()


def test_send_conversation_success():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_chat_response

        client = LLMClient(token="fake-token")
        messages = [{"role": "user", "content": "What's up?"}]
        result = client.send_conversation(messages)

        assert result == "Hello from the LLM!"


def test_get_models_failure_logs(capfd):
    with patch("requests.get", side_effect=requests.RequestException("Boom")):
        client = LLMClient(token="fail")
        result = client.get_models()

        out, _ = capfd.readouterr()
        assert result is None
        assert "Request failed: Boom" in out
