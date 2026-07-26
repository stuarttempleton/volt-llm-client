from unittest.mock import MagicMock, Mock, patch
import importlib.util
import os
import pytest
import requests
from voltllmclient.client import LLMClient
from voltllmclient.mcptools import MCPToolProvider

# Only connect() needs fastmcp; everything else runs on a bare install.
needs_fastmcp = pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None,
    reason="fastmcp not installed (pip install volt-llm-client[mcp])"
)


def _resp(json_data, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = lambda: None
    resp.text = str(json_data)
    return resp


def _ollama_get(url, *args, **kwargs):
    if url.endswith("/api/models"):
        return _resp([])
    return _resp({"models": [{"name": "gemma4"}]})


def _mock_mcp(result="tool result"):
    """A stand-in for MCPToolProvider - no Docker or fastmcp needed."""
    mcp = Mock()
    mcp.tool_specs.return_value = [{
        "type": "function",
        "function": {"name": "get_file", "description": "Read a file.", "parameters": {"type": "object"}}
    }]
    mcp.call.return_value = result
    return mcp


def _tool_call_response(api_type, args):
    call = {"function": {"name": "get_file", "arguments": args}}
    if api_type == "openwebui":
        call["id"] = "call_abc123"
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}
    return {"message": {"role": "assistant", "tool_calls": [call]}}


# --- extract_message -------------------------------------------------------

def test_extract_message_openwebui():
    with patch("requests.get", return_value=_resp({"data": []})):
        client = LLMClient(token="t")
        msg = client.extract_message({"choices": [{"message": {"content": "hi", "tool_calls": []}}]})
        assert msg["content"] == "hi"


def test_extract_message_ollama():
    with patch("requests.get", side_effect=_ollama_get):
        client = LLMClient(token="t")
        assert client.extract_message({"message": {"content": "hi"}})["content"] == "hi"


def test_extract_message_unknown_sniffs_both():
    with patch("requests.get", side_effect=requests.RequestException("down")):
        client = LLMClient(token="t")
        assert client.api_type == "unknown"
        assert client.extract_message({"choices": [{"message": {"content": "a"}}]})["content"] == "a"
        assert client.extract_message({"message": {"content": "b"}})["content"] == "b"
        assert client.extract_message({"nonsense": 1}) == {}


# --- _normalize_tool_calls -------------------------------------------------

def test_normalize_tool_calls_ollama_dict_args():
    with patch("requests.get", side_effect=_ollama_get):
        client = LLMClient(token="t")
        calls = client._normalize_tool_calls(
            {"tool_calls": [{"function": {"name": "get_file", "arguments": {"id": "notes.txt"}}}]})
        assert calls == [(None, "get_file", {"id": "notes.txt"})]


def test_normalize_tool_calls_openwebui_string_args():
    with patch("requests.get", return_value=_resp({"data": []})):
        client = LLMClient(token="t")
        calls = client._normalize_tool_calls({"tool_calls": [
            {"id": "call_1", "function": {"name": "get_file", "arguments": '{"id": "notes.txt"}'}}]})
        assert calls == [("call_1", "get_file", {"id": "notes.txt"})]


def test_normalize_tool_calls_bad_json_degrades(capfd):
    with patch("requests.get", return_value=_resp({"data": []})):
        client = LLMClient(token="t")
        calls = client._normalize_tool_calls({"tool_calls": [
            {"id": "c", "function": {"name": "get_file", "arguments": "{not json"}}]})
        assert calls == [("c", "get_file", {})]
        assert "Could not parse tool arguments" in capfd.readouterr()[0]


def test_normalize_tool_calls_empty():
    with patch("requests.get", return_value=_resp({"data": []})):
        client = LLMClient(token="t")
        assert client._normalize_tool_calls({"content": "no tools here"}) == []
        assert client._normalize_tool_calls(None) == []


# --- send_with_tools -------------------------------------------------------

def test_send_with_tools_openwebui_round_trip():
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}')),
            _resp({"choices": [{"message": {"content": "The file says hello."}}]}),
        ]
        mcp = _mock_mcp("hello")
        client = LLMClient(token="t", mcp=mcp)
        result = client.send_with_tools([{"role": "user", "content": "what is in my file?"}])

        assert result == "The file says hello."
        mcp.call.assert_called_once_with("get_file", {"id": "notes.txt"})

        second = mock_post.call_args_list[1].kwargs["json"]["messages"]
        tool_msg = [m for m in second if m["role"] == "tool"][0]
        assert tool_msg["content"] == "hello"
        assert tool_msg["tool_call_id"] == "call_abc123"
        assert second[0]["role"] == "user"
        # tools advertised on every round
        assert mock_post.call_args_list[1].kwargs["json"]["tools"] == mcp.tool_specs.return_value


def test_send_with_tools_ollama_omits_tool_call_id():
    with patch("requests.get", side_effect=_ollama_get), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("ollama", {"id": "notes.txt"})),
            _resp({"message": {"content": "Done."}}),
        ]
        mcp = _mock_mcp()
        client = LLMClient(token="t", mcp=mcp)
        result = client.send_with_tools([{"role": "user", "content": "go"}])

        assert result == "Done."
        tool_msg = [m for m in mock_post.call_args_list[1].kwargs["json"]["messages"]
                    if m["role"] == "tool"][0]
        assert "tool_call_id" not in tool_msg
        assert tool_msg["name"] == "get_file"


def test_send_with_tools_no_mcp_is_single_round():
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", return_value=_resp({"choices": [{"message": {"content": "plain"}}]})) as mock_post:
        client = LLMClient(token="t")
        result = client.send_with_tools([{"role": "user", "content": "hi"}])

        assert result == "plain"
        assert mock_post.call_count == 1
        assert "tools" not in mock_post.call_args.kwargs["json"]


def test_send_with_tools_does_not_mutate_caller_messages():
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}')),
            _resp({"choices": [{"message": {"content": "done"}}]}),
        ]
        client = LLMClient(token="t", mcp=_mock_mcp())
        messages = [{"role": "user", "content": "hi"}]
        client.send_with_tools(messages)

        assert messages == [{"role": "user", "content": "hi"}]


def test_send_with_tools_collects_transcript():
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}')),
            _resp({"choices": [{"message": {"content": "done"}}]}),
        ]
        client = LLMClient(token="t", mcp=_mock_mcp("hello"))
        transcript = []
        client.send_with_tools([{"role": "user", "content": "hi"}], transcript=transcript)

        # The tool_calls message and its result, in order, and nothing else.
        assert [m["role"] for m in transcript] == ["assistant", "tool"]
        assert transcript[0]["tool_calls"][0]["function"]["name"] == "get_file"
        assert transcript[1]["content"] == "hello"
        assert transcript[1]["tool_call_id"] == "call_abc123"


def test_send_with_tools_transcript_empty_when_no_tools_used():
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", return_value=_resp({"choices": [{"message": {"content": "plain"}}]})):
        client = LLMClient(token="t", mcp=_mock_mcp())
        transcript = []
        assert client.send_with_tools([{"role": "user", "content": "hi"}], transcript=transcript) == "plain"
        assert transcript == []


def test_send_with_tools_transcript_is_optional():
    # Omitting transcript must keep the old behaviour exactly.
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}')),
            _resp({"choices": [{"message": {"content": "done"}}]}),
        ]
        client = LLMClient(token="t", mcp=_mock_mcp())
        assert client.send_with_tools([{"role": "user", "content": "hi"}]) == "done"


def test_send_with_tools_caps_rounds(capfd):
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        # Model asks for a tool forever
        mock_post.return_value = _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}'))
        mcp = _mock_mcp()
        client = LLMClient(token="t", mcp=mcp)
        result = client.send_with_tools([{"role": "user", "content": "loop"}], max_tool_rounds=2)

        assert mock_post.call_count == 3  # 2 rounds + final
        assert mcp.call.call_count == 3
        assert result == ""
        assert "Stopped after 2 tool rounds" in capfd.readouterr()[0]


def test_send_with_tools_request_failure_logs(capfd):
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", side_effect=requests.RequestException("Boom")):
        client = LLMClient(token="t", mcp=_mock_mcp())
        result = client.send_with_tools([{"role": "user", "content": "hi"}])

        assert result is None
        assert "Request failed: Boom" in capfd.readouterr()[0]


def test_send_with_tools_timeout_hints_at_tool_count(capfd):
    with patch("requests.get", return_value=_resp({"data": []})), \
         patch("requests.post", side_effect=requests.Timeout("too slow")):
        mcp = _mock_mcp()
        mcp.tool_specs.return_value = [
            {"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}} for i in range(93)
        ]
        client = LLMClient(token="t", mcp=mcp)
        result = client.send_with_tools([{"role": "user", "content": "hi"}])

        out = capfd.readouterr()[0]
        assert result is None
        assert "93 tools advertised" in out
        assert "Narrow the tool list" in out


def test_send_with_tools_tool_error_is_fed_back():
    with patch("requests.get", return_value=_resp({"data": []})), patch("requests.post") as mock_post:
        mock_post.side_effect = [
            _resp(_tool_call_response("openwebui", '{"id": "notes.txt"}')),
            _resp({"choices": [{"message": {"content": "I could not fetch it."}}]}),
        ]
        mcp = _mock_mcp("Error: gateway exploded")
        client = LLMClient(token="t", mcp=mcp)
        result = client.send_with_tools([{"role": "user", "content": "hi"}])

        tool_msg = [m for m in mock_post.call_args_list[1].kwargs["json"]["messages"]
                    if m["role"] == "tool"][0]
        assert tool_msg["content"] == "Error: gateway exploded"
        assert result == "I could not fetch it."


# --- MCPToolProvider filtering (no gateway needed) -------------------------

def _provider(**kwargs):
    p = MCPToolProvider(**kwargs)
    p.close()  # we only exercise pure filtering; stop the loop thread immediately
    return p


def test_filter_defaults_to_everything():
    p = _provider()
    assert p._keep("get_file") is True
    assert p._keep("write_file") is True


def test_filter_exact_tool_names():
    p = _provider(tools=["get_file", "list_dirs"])
    assert p._keep("get_file") is True
    assert p._keep("write_file") is False


def test_filter_glob_include():
    p = _provider(include="get_*, search_*")
    assert p._keep("get_file") is True
    assert p._keep("search_files") is True
    assert p._keep("write_file") is False


def test_profile_is_in_gateway_args():
    p = _provider(profile="my_profile")
    assert p.command == "docker"
    assert p.args == ["mcp", "gateway", "run", "--profile", "my_profile"]


def test_connect_without_profile_refuses(capfd):
    # A profile-less gateway serves only its own meta-tools (mcp-add, mcp-config-set, ...),
    # which would let a model rewrite the user's MCP setup. Refuse instead of spawning it.
    p = MCPToolProvider()
    try:
        assert p.connect() is False
        assert p._client is None
        assert "MCP not configured" in capfd.readouterr()[0]
    finally:
        p.close()


def test_custom_args_connect_without_a_profile():
    # args= means a non-Docker stdio server, where profiles are meaningless.
    p = _provider(args=["serve", "--stdio"], command="my-mcp")
    assert p.command == "my-mcp"
    assert p.args == ["serve", "--stdio"]
    assert p._configured is True


def test_tool_specs_empty_when_not_connected():
    p = _provider()
    assert p.tool_specs() == []


def test_call_without_connection_returns_error_string():
    p = _provider()
    assert p.call("get_file", {}) == "Error: MCP is not connected."


def test_tool_specs_builds_openai_shape():
    p = MCPToolProvider(include="get_*")
    try:
        tool = Mock()
        tool.name = "get_file"
        tool.description = "Read a file."
        tool.inputSchema = {"type": "object", "properties": {"id": {"type": "string"}}}
        skipped = Mock()
        skipped.name = "write_file"
        skipped.description = "nope"
        skipped.inputSchema = {}
        p._client = Mock()
        with patch.object(p, "_run", return_value=[tool, skipped]):
            specs = p.tool_specs()

        assert specs == [{
            "type": "function",
            "function": {
                "name": "get_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}
            }
        }]
    finally:
        p.close()


def test_tool_specs_warns_when_filter_matches_nothing(capfd):
    p = MCPToolProvider(include="typo_*")
    try:
        tool = Mock()
        tool.name = "get_file"
        tool.description = ""
        tool.inputSchema = {}
        p._client = Mock()
        with patch.object(p, "_run", return_value=[tool]):
            assert p.tool_specs() == []
        # Silently advertising nothing looks identical to "the model ignored the tools".
        assert "matched none of 1 tools" in capfd.readouterr()[0]
    finally:
        p.close()


def test_tool_specs_silent_on_success(capfd):
    p = MCPToolProvider(include="get_*")
    try:
        tool = Mock()
        tool.name = "get_file"
        tool.description = ""
        tool.inputSchema = {}
        p._client = Mock()
        with patch.object(p, "_run", return_value=[tool]):
            assert len(p.tool_specs()) == 1
        assert capfd.readouterr()[0] == ""
    finally:
        p.close()


def test_call_flattens_text_content():
    p = MCPToolProvider()
    try:
        block = Mock()
        block.text = "line one"
        block2 = Mock()
        block2.text = "line two"
        result = Mock()
        result.content = [block, block2]
        p._client = Mock()
        with patch.object(p, "_run", return_value=result):
            assert p.call("get_file", {"id": "notes.txt"}) == "line one\nline two"
    finally:
        p.close()


def test_call_returns_error_string_on_exception(capfd):
    p = MCPToolProvider()
    try:
        p._client = Mock()
        with patch.object(p, "_run", side_effect=RuntimeError("gateway died")):
            assert p.call("get_file", {}) == "Error: gateway died"
        assert "failed" in capfd.readouterr()[0]
    finally:
        p.close()


GATEWAY_PANIC = """panic: unable to get 'ProgramData' [recovered, repanicked]

goroutine 1 [running]:
github.com/docker/mcp-gateway/pkg/desktop.init.OnceValue[...].func2.1.1()
\tsync/oncefunc.go:63 +0x75
panic({0x7ff6843e69a0?, 0xc0003a2ce0?})
\truntime/panic.go:783 +0x132
main.main()
"""


def _provider_with_log(tmp_path, contents):
    log = tmp_path / "gw.log"
    log.write_text(contents, encoding="utf-8")
    return MCPToolProvider(profile="my_profile", log_file=str(log))


def test_gateway_error_extracts_panic_cause(tmp_path):
    p = _provider_with_log(tmp_path, GATEWAY_PANIC)
    try:
        assert p._gateway_error() == "panic: unable to get 'ProgramData'"
    finally:
        p.close()


def test_gateway_error_extracts_plain_message(tmp_path):
    p = _provider_with_log(tmp_path, "Docker Desktop is not running\n")
    try:
        assert p._gateway_error() == "Docker Desktop is not running"
    finally:
        p.close()


def test_gateway_error_skips_progress_noise(tmp_path):
    log = "- Reading profile configuration...\n> 85 tools listed\nprofile bogus not found\n"
    p = _provider_with_log(tmp_path, log)
    try:
        assert p._gateway_error() == "profile bogus not found"
    finally:
        p.close()


def test_gateway_error_missing_log_returns_none(tmp_path):
    p = MCPToolProvider(log_file=str(tmp_path / "does-not-exist.log"))
    try:
        assert p._gateway_error() is None
    finally:
        p.close()


def test_gateway_log_defaults_to_unique_tempfile():
    a, b = MCPToolProvider(), MCPToolProvider()
    try:
        assert a._log_path.endswith(".log") and "volt-mcp-gateway-" in a._log_path
        # Concurrent providers must not share a log, or they truncate each other's output.
        assert a._log_path != b._log_path
        assert os.path.exists(a._log_path) and os.path.exists(b._log_path)
    finally:
        a.close()
        b.close()


def test_own_log_removed_on_clean_close():
    p = MCPToolProvider()
    path = p._log_path
    p.close()
    assert not os.path.exists(path)


@needs_fastmcp
def test_own_log_kept_after_failed_connect():
    p = MCPToolProvider(profile="my_profile")
    path = p._log_path
    try:
        with patch("fastmcp.Client", return_value=MagicMock()), \
             patch.object(p, "_run", side_effect=RuntimeError("boom")):
            assert p.connect() is False
    finally:
        p.close()
    # The warning tells the user to read this file, so it must survive close().
    assert os.path.exists(path)
    os.unlink(path)


def test_supplied_log_file_is_not_deleted(tmp_path):
    log = tmp_path / "mine.log"
    log.write_text("keep me", encoding="utf-8")
    p = MCPToolProvider(log_file=str(log))
    p.close()
    assert log.read_text(encoding="utf-8") == "keep me"


@needs_fastmcp
def test_connect_reports_gateway_error_over_exception(tmp_path, capfd):
    p = _provider_with_log(tmp_path, "Docker Desktop is not running\n")
    try:
        with patch("fastmcp.Client", return_value=MagicMock()), \
             patch.object(p, "_run", side_effect=RuntimeError("Connection closed")), \
             patch.object(p, "_gateway_error", return_value="Docker Desktop is not running"):
            assert p.connect() is False
        out = capfd.readouterr()[0]
        # The gateway's own reason beats the opaque transport error.
        assert "Docker Desktop is not running" in out
        assert "Connection closed" not in out
    finally:
        p.close()


@needs_fastmcp
def test_connect_degrades_when_gateway_unavailable(tmp_path, capfd):
    # Supply a log path so the failure log lands in tmp_path instead of the real temp dir.
    p = MCPToolProvider(profile="my_profile", log_file=str(tmp_path / "gw.log"))
    try:
        # Mock out fastmcp.Client so no real coroutine is created, then fail the connect.
        with patch("fastmcp.Client", return_value=MagicMock()), \
             patch.object(p, "_run", side_effect=RuntimeError("Docker Desktop is not running")):
            assert p.connect() is False
        assert p._client is None
        assert "MCP unavailable" in capfd.readouterr()[0]
    finally:
        p.close()


@needs_fastmcp
def test_connect_succeeds_and_caches_session():
    p = MCPToolProvider(profile="my_profile")
    try:
        with patch("fastmcp.Client", return_value=MagicMock()), \
             patch.object(p, "_run", return_value=None), \
             patch.object(p, "tool_specs", return_value=[{"type": "function"}]):
            assert p.connect() is True
        assert p._client is not None
    finally:
        p.close()
