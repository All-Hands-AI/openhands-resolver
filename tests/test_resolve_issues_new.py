import os
import tempfile
import pytest


from unittest.mock import AsyncMock, patch, MagicMock
from openhands_resolver.issue_definitions import IssueHandler, PRHandler
from openhands_resolver.resolve_issue import (
    initialize_runtime,
    complete_runtime,
    process_issue,
)
from openhands_resolver.github_issue import GithubIssue
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, NullObservation
from openhands_resolver.resolver_output import ResolverOutput
from openhands.core.config import LLMConfig
from openhands.events.stream import EventStream


def test_guess_success():
    mock_issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
    )
    mock_history = [
        MagicMock(message="Test message")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="--- success\ntrue\n--- explanation\nIssue resolved successfully"))]
    issue_handler = IssueHandler("owner", "repo", "token")

    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, comment_success, explanation = issue_handler.guess_success(mock_issue, mock_history, mock_llm_config)
        assert issue_handler.issue_type == "issue"
        assert comment_success is None
        assert success
        assert explanation == "Issue resolved successfully"

def test_guess_success_with_thread_comments():
    mock_issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
        thread_comments=["First comment", "Second comment", "latest feedback:\nPlease add tests"]
    )
    mock_history = [
        MagicMock(message="I have added tests for this case")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="--- success\ntrue\n--- explanation\nTests have been added to verify thread comments handling"))]
    issue_handler = IssueHandler("owner", "repo", "token")

    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, comment_success, explanation = issue_handler.guess_success(mock_issue, mock_history, mock_llm_config)
        assert issue_handler.issue_type == "issue"
        assert comment_success is None
        assert success
        assert "Tests have been added" in explanation

def test_guess_success_failure():
    mock_issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
        thread_comments=["First comment", "Second comment", "latest feedback:\nPlease add tests"]
    )
    mock_history = [
        MagicMock(message="I have added tests for this case")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="--- success\ntrue\n--- explanation\nTests have been added to verify thread comments handling"))]
    issue_handler = IssueHandler("owner", "repo", "token")

    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, comment_success, explanation = issue_handler.guess_success(mock_issue, mock_history, mock_llm_config)
        assert issue_handler.issue_type == "issue"
        assert comment_success is None
        assert success
        assert "Tests have been added" in explanation

def test_guess_success_negative_case():
    mock_issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
    )
    mock_history = [
        MagicMock(message="Failed to fix the issue")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="--- success\nfalse\n--- explanation\nIssue not resolved"))]    
    issue_handler = IssueHandler("owner", "repo", "token")

    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, comment_success, explanation = issue_handler.guess_success(mock_issue, mock_history, mock_llm_config)
        assert issue_handler.issue_type == "issue"
        assert comment_success is None
        assert not success
        assert explanation == "Issue not resolved"

def test_guess_success_invalid_output():
    mock_issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
    )
    mock_history = [
        MagicMock(message="Test message")
    ]
    mock_llm_config = LLMConfig(model="test_model", api_key="test_api_key")

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock(message=MagicMock(content="This is not a valid output"))]
    issue_handler = IssueHandler("owner", "repo", "token")

    with patch('litellm.completion', MagicMock(return_value=mock_completion_response)):
        success, comment_success, explanation = issue_handler.guess_success(mock_issue, mock_history, mock_llm_config)
        assert issue_handler.issue_type == "issue"
        assert comment_success is None
        assert not success
        assert explanation == "Failed to decode answer from LLM response: This is not a valid output"