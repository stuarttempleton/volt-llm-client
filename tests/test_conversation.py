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

        MockProvider.assert_called_once_with(profile="my_profile")
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

