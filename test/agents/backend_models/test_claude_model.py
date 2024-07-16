import pytest

from crab import MessageType, action
from crab.agents.backend_models.claude_model import ClaudeModel

# TODO: Add mock data

@pytest.fixture
def claude_model_text():
    return ClaudeModel(
        model="claude-3-opus-20240229",
        parameters={"max_tokens": 3000},
        history_messages_len=1,
    )


@action
def add(a: int, b: int):
    """Add up two integers.

    Args:
        a: An addend
        b: Another addend
    """
    return a + b

@pytest.mark.skip(reason="Mock data to be added")
def test_text_chat(claude_model_text):
    message = ("Hello!", MessageType.TEXT)
    output = claude_model_text.chat(message)
    assert output.message
    assert output.action_list is None
    assert claude_model_text.token_usage > 0

    # Send another message to check accumulated tokens and history length
    message2 = ("Give me five!", MessageType.TEXT)
    output = claude_model_text.chat(message2)
    assert claude_model_text.token_usage > 0
    assert output.message
    assert len(claude_model_text.chat_history) == 2

    # Send another message to check accumulated tokens and chat history
    output = claude_model_text.chat(message2)
    assert output.message
    assert len(claude_model_text.chat_history) == 3


@pytest.mark.skip(reason="Mock data to be added")
def test_action_chat(claude_model_text):
    claude_model_text.reset("You are a helpful assistant.", [add])
    message = (
        "I had 10 dollars. Miss Polaris gave me 15 dollars. How many money do I have now.",
        0,
    )
    output = claude_model_text.chat(message)
    assert output.message is None
    assert len(output.action_list) == 1
    assert output.action_list[0].arguments == {"a": 10, "b": 15}
    assert output.action_list[0].name == "add"
    assert claude_model_text.token_usage > 0
