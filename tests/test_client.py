from unittest.mock import Mock
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
        def get_side_effect(url, *args, **kwargs):
            if url.endswith("/api/models"):
                resp = Mock()
                resp.status_code = 200
                resp.json.return_value = mock_model_response
                resp.raise_for_status = lambda: None
                return resp
            else:
                resp = Mock()
                resp.status_code = 200
                resp.json.return_value = {"data": []}
                resp.raise_for_status = lambda: None
                return resp
        mock_get.side_effect = get_side_effect

        client = LLMClient(token="fake-token")
        result = client.get_models()

        assert result == mock_model_response


def test_send_prompt_success():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # API detection
        detection_resp = type("Resp", (), {
            "status_code": 200,
            "json": lambda: {"data": []},
            "raise_for_status": lambda: None
        })()
        mock_get.side_effect = [detection_resp]
        # Chat completion
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_chat_response

        client = LLMClient(token="fake-token")
        result = client.send_prompt("Hello")

        assert result == "Hello from the LLM!"


def test_send_conversation_success():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # API detection
        detection_resp = type("Resp", (), {
            "status_code": 200,
            "json": lambda: {"data": []},
            "raise_for_status": lambda: None
        })()
        mock_get.side_effect = [detection_resp]
        # Chat completion
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_chat_response

        client = LLMClient(token="fake-token")
        messages = [{"role": "user", "content": "What's up?"}]
        result = client.send_conversation(messages)

        assert result == "Hello from the LLM!"


def test_get_models_failure_logs(capfd):
    with patch("requests.get") as mock_get:
        call_count = {"models": 0}
        def get_side_effect(url, *args, **kwargs):
            if url.endswith("/api/models"):
                call_count["models"] += 1
                if call_count["models"] == 2:
                    raise requests.RequestException("Boom")
                else:
                    resp = Mock()
                    resp.status_code = 200
                    resp.json.return_value = {"data": []}
                    resp.raise_for_status = lambda: None
                    return resp
            else:
                resp = Mock()
                resp.status_code = 200
                resp.json.return_value = {"data": []}
                resp.raise_for_status = lambda: None
                return resp
        mock_get.side_effect = get_side_effect

        client = LLMClient(token="fail")
        result = client.get_models()

        out, _ = capfd.readouterr()
        assert result is None
        assert "Request failed: Boom" in out
