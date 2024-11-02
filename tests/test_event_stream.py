import os
import pytest
from unittest.mock import MagicMock, patch
from openhands.core.config import LLMConfig
from openhands.events.stream import EventStreamSubscriber
from openhands.events.observation import CmdOutputObservation
from openhands_resolver.resolve_issue import process_issue
from openhands_resolver.github_issue import GithubIssue
from openhands_resolver.issue_definitions import IssueHandler

@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.event_stream = MagicMock()
    runtime.event_stream.subscribe = MagicMock()
    
    # Make connect() async-compatible
    async def mock_connect():
        return None
    runtime.connect = mock_connect
    
    # Make run_action async-compatible
    async def mock_run_action(action):
        return CmdOutputObservation(content="", exit_code=0)
    runtime.run_action = mock_run_action
    
    return runtime

@pytest.fixture
def mock_issue():
    return GithubIssue(
        number=123,
        title="Test Issue",
        body="Test body",
        labels=["bug"],
        state="open",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        html_url="https://github.com/test/test/issues/123",
        user="test-user",
        assignees=[],
        milestone=None,
        comments=[],
        head_branch=None,
        base_branch=None,
        review_comments=[],
        owner="test",
        repo="test",
    )

@pytest.mark.asyncio
async def test_event_stream_subscription(mock_runtime, mock_issue, tmp_path):
    with patch("openhands_resolver.resolve_issue.create_runtime", return_value=mock_runtime):
        # Create test output directory
        output_dir = str(tmp_path)
        os.makedirs(os.path.join(output_dir, "repo"), exist_ok=True)
        
        # Mock issue handler
        issue_handler = IssueHandler("test", "test", "test-token")
        
        # Run process_issue
        await process_issue(
            issue=mock_issue,
            base_commit="test-commit",
            max_iterations=1,
            llm_config=LLMConfig(model="test-model", api_key="test-key"),
            output_dir=output_dir,
            runtime_container_image="test-image",
            prompt_template="test-template",
            issue_handler=issue_handler,
            repo_instruction=None,
            reset_logger=True,
        )
        
        # Verify event stream subscription was called
        mock_runtime.event_stream.subscribe.assert_called_once()
        args = mock_runtime.event_stream.subscribe.call_args
        assert args[0][1] == EventStreamSubscriber.MAIN  # Check subscriber type
        assert callable(args[0][0])  # Check event handler is callable
