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


def _resp(json_data, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = lambda: None
    resp.text = str(json_data)
    return resp


def _ollama_get(url, *args, **kwargs):
    # /api/models must not look like Open WebUI, so detection falls through to /api/tags
    if url.endswith("/api/models"):
        return _resp([])
    return _resp({"models": [{"name": "gemma3"}]})


def test_detect_api_type_ollama():
    with patch("requests.get", side_effect=_ollama_get):
        client = LLMClient(token="fake-token")

        assert client.api_type == "ollama"
        assert client.endpoints["chat"].endswith("/api/chat")
        assert client.endpoints["models"].endswith("/api/tags")


def test_ollama_payload_nests_temperature_in_options():
    with patch("requests.get", side_effect=_ollama_get), patch("requests.post") as mock_post:
        mock_post.return_value = _resp({"message": {"content": "Hello from Ollama!"}})

        client = LLMClient(token="fake-token", temperature=0.7)
        result = client.send_prompt("Hello")

        payload = mock_post.call_args.kwargs["json"]
        assert result == "Hello from Ollama!"
        assert payload["options"] == {"temperature": 0.7}
        assert "temperature" not in payload


def test_openwebui_payload_keeps_temperature_top_level():
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.return_value = _resp(mock_chat_response)

        client = LLMClient(token="fake-token", temperature=0.7)
        client.send_conversation([{"role": "user", "content": "Hello"}])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.7
        assert "options" not in payload


def test_timeout_passed_to_requests():
    with patch("requests.get", return_value=_resp({"data": []})) as mock_get, \
         patch("requests.post", return_value=_resp(mock_chat_response)) as mock_post:
        client = LLMClient(token="fake-token", timeout=7)

        client.get_models()
        client.send_prompt("Hello")
        client.send_conversation([{"role": "user", "content": "Hello"}])

        # Last GET is get_models (earlier ones are API detection, which uses its own short timeout)
        assert mock_get.call_args.kwargs["timeout"] == 7
        for call in mock_post.call_args_list:
            assert call.kwargs["timeout"] == 7


def test_send_conversation_failure_logs(capfd):
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", side_effect=requests.RequestException("Boom")):
        client = LLMClient(token="fake-token")
        result = client.send_conversation([{"role": "user", "content": "Hello"}])

        out, _ = capfd.readouterr()
        assert result is None
        assert "Request failed: Boom" in out


def test_send_conversation_unexpected_response_logs(capfd):
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", return_value=_resp({"unexpected": "shape"})):
        client = LLMClient(token="fake-token")
        result = client.send_conversation([{"role": "user", "content": "Hello"}])

        out, _ = capfd.readouterr()
        assert result is None
        assert "Unexpected response" in out


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
