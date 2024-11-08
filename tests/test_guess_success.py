import pytest
from openhands_resolver.issue_definitions import IssueHandler
from openhands_resolver.github_issue import GithubIssue
from openhands.events.action.message import MessageAction
from openhands.core.config import LLMConfig

def test_guess_success_multiline_explanation():
    # Mock data
    issue = GithubIssue(
        owner="test",
        repo="test",
        number=1,
        title="Test Issue",
        body="Test body",
        thread_comments=None,
        review_comments=None
    )
    history = [MessageAction(content="Test message")]
    llm_config = LLMConfig(model="test", api_key="test")
    
    # Create a mock response with multi-line explanation
    mock_response = """--- success
true

--- explanation
The PR successfully addressed the issue by:
- Fixed bug A
- Added test B
- Updated documentation C

Automatic fix generated by OpenHands 🙌"""
    
    # Create a handler instance
    handler = IssueHandler("test", "test", "test")
    
    # Mock the litellm.completion call
    def mock_completion(*args, **kwargs):
        class MockResponse:
            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content
                def __init__(self, content):
                    self.message = self.Message(content)
            def __init__(self, content):
                self.choices = [self.Choice(content)]
        return MockResponse(mock_response)
    
    # Patch the litellm.completion function
    import litellm
    original_completion = litellm.completion
    litellm.completion = mock_completion
    
    try:
        # Call guess_success
        success, _, explanation = handler.guess_success(issue, history, llm_config)
        
        # Verify the results
        assert success is True
        assert "The PR successfully addressed the issue by:" in explanation
        assert "Fixed bug A" in explanation
        assert "Added test B" in explanation
        assert "Updated documentation C" in explanation
        assert "Automatic fix generated by OpenHands" in explanation
    finally:
        # Restore the original function
        litellm.completion = original_completion
