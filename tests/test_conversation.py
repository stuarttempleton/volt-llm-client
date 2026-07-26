import json
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock
from voltllmclient.conversation import LLMConversation

@pytest.fixture
def mock_client():
    with patch("voltllmclient.conversation.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.send_conversation.return_value = "Mocked response"
        yield instance

@pytest.fixture
def conversation(mock_client):
    return LLMConversation(model="mock-model", system_prompt="You are mock AI.")

def test_send_default(conversation, mock_client):
    response = conversation.send("Hello!")
    assert response == "Mocked response"
    assert len(conversation.messages) == 3  # system, user, assistant
    assert conversation.messages[-1]["role"] == "assistant"

def test_send_with_full_context(conversation, mock_client):
    conversation.messages.append({"role": "assistant", "content": "Previous response"})
    conversation.send_with_full_context("Follow-up question")

    args = mock_client.send_conversation.call_args[0][0]
    assert isinstance(args, list)
    assert any(m["role"] == "assistant" for m in args)

def test_send_with_summary_context(conversation, mock_client):
    conversation.messages.append({"role": "assistant", "content": "Summary point"})
    conversation.send_with_summary_context("And now?")

    args = mock_client.send_conversation.call_args[0][0]
    roles = [m["role"] for m in args]
    assert "assistant" in roles
    assert "system" in roles
    assert roles.count("user") == 1  # Only new user message

# --- mcp lifecycle ---------------------------------------------------------

def test_profile_string_builds_and_owns_a_provider(mock_client):
    with patch("voltllmclient.mcptools.MCPToolProvider") as MockProvider:
        provider = MockProvider.return_value
        provider.connect.return_value = True
        conv = LLMConversation(mcp="my_profile")

        MockProvider.assert_called_once_with(profile="my_profile", include=None)
        assert conv.use_tools is True
        conv.close()
        provider.close.assert_called_once()


def test_supplied_provider_is_not_closed(mock_client):
    provider = MagicMock()
    conv = LLMConversation(mcp=provider)

    assert conv.use_tools is True
    conv.close()
    # The caller owns this one; closing it out from under them would be rude.
    provider.close.assert_not_called()


def test_failed_connect_leaves_a_working_chat(mock_client):
    with patch("voltllmclient.mcptools.MCPToolProvider") as MockProvider:
        MockProvider.return_value.connect.return_value = False
        conv = LLMConversation(mcp="my_profile")

        assert conv.use_tools is False
        assert conv.send("Hello!") == "Mocked response"
        conv.close()


def test_no_mcp_never_imports_a_provider(mock_client):
    with patch("voltllmclient.mcptools.MCPToolProvider") as MockProvider:
        conv = LLMConversation()
        assert conv.use_tools is False
        conv.close()
        MockProvider.assert_not_called()


def test_context_manager_closes_own_provider(mock_client):
    with patch("voltllmclient.mcptools.MCPToolProvider") as MockProvider:
        provider = MockProvider.return_value
        provider.connect.return_value = True
        with LLMConversation(mcp="my_profile") as conv:
            assert conv.use_tools is True
        provider.close.assert_called_once()


# --- tool history retention ------------------------------------------------

def _tool_using_conversation(mock_client, tool_msgs):
    """A conversation whose client reports a tool round-trip via transcript=."""
    def fake_send(prompt, transcript=None, **kwargs):
        if transcript is not None:
            transcript.extend(tool_msgs)
        return "Answer from tool"
    mock_client.send_with_tools.side_effect = fake_send

    provider = MagicMock()
    return LLMConversation(mcp=provider, system_prompt="You are mock AI.")


def test_tool_round_trip_is_kept_in_history(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "user found: usr_123"},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.send("who is tupper?")

    roles = [m["role"] for m in conv.messages]
    # system, user, assistant(tool_calls), tool, assistant(answer)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert any(m["role"] == "tool" and "usr_123" in m["content"] for m in conv.messages)
    # The tool evidence must sit after the user turn and before the final answer.
    assert conv.messages[-1] == {"role": "assistant", "content": "Answer from tool"}


def test_tool_history_is_replayed_on_the_next_turn(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "user found: usr_123"},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.send("who is tupper?")
    mock_client.send_with_tools.side_effect = lambda prompt, transcript=None, **kw: "Second answer"
    conv.send_with_full_context("what groups are they in?")

    # Turn 2 must see the tool result from turn 1, or it re-runs the lookup.
    sent = mock_client.send_with_tools.call_args[0][0]
    assert any(m["role"] == "tool" and "usr_123" in m["content"] for m in sent)


def test_summary_context_skips_orphaned_tool_calls(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "user found: usr_123"},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.send("who is tupper?")
    mock_client.send_with_tools.side_effect = lambda prompt, transcript=None, **kw: "Second answer"
    conv.send_with_summary_context("and now?")

    sent = mock_client.send_with_tools.call_args[0][0]
    # assistant_only drops role="tool", so a tool_calls message would be orphaned
    # and rejected by OpenAI-compatible endpoints.
    assert not any(m.get("tool_calls") for m in sent)
    assert not any(m["role"] == "tool" for m in sent)


def test_large_tool_result_is_truncated_in_history(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user_groups", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 50_000},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.max_tool_result_chars = 4000
    conv.send("what groups?")

    stored = [m for m in conv.messages if m["role"] == "tool"][0]
    assert len(stored["content"]) < 4200
    assert "46000 chars truncated" in stored["content"]
    # The original transcript entry must not be mutated in place.
    assert len(tool_msgs[1]["content"]) == 50_000


def test_small_tool_result_is_kept_verbatim(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "usr_123"},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.send("who?")
    assert [m for m in conv.messages if m["role"] == "tool"][0]["content"] == "usr_123"


def test_tool_results_are_kept_whole_by_default(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "y" * 20_000},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    assert conv.max_tool_result_chars is None
    conv.send("dump it")
    assert len([m for m in conv.messages if m["role"] == "tool"][0]["content"]) == 20_000


def test_mcp_include_is_passed_to_provider(mock_client):
    with patch("voltllmclient.mcptools.MCPToolProvider") as MockProvider:
        MockProvider.return_value.connect.return_value = True
        conv = LLMConversation(mcp="my_profile", mcp_include="get_*,search_*")
        MockProvider.assert_called_once_with(profile="my_profile", include="get_*,search_*")
        conv.close()


def test_tool_history_survives_transcript_round_trip(mock_client):
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "user found: usr_123"},
    ]
    conv = _tool_using_conversation(mock_client, tool_msgs)
    conv.send("who is tupper?")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_file.close()
    try:
        conv.save_transcript(tmp_file.name)
        reloaded = LLMConversation()
        reloaded.load_transcript(tmp_file.name)
        assert reloaded.messages == conv.messages
    finally:
        os.unlink(tmp_file.name)


def test_save_and_load_transcript(conversation):
    conversation.send("Testing save/load")

    # Create temp file and close it immediately so Windows allows re-opening
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_file.close()

    try:
        conversation.save_transcript(tmp_file.name)

        # New conversation instance to test loading
        new_conv = LLMConversation()
        new_conv.load_transcript(tmp_file.name)

        assert new_conv.messages == conversation.messages

    finally:
        os.unlink(tmp_file.name)  # Always clean up

