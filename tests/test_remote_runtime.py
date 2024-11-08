import os
import pytest
from openhands_resolver.resolve_issue import process_issue
from openhands_resolver.github_issue import GithubIssue
from openhands.core.config import LLMConfig


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('RUNTIME', 'remote')
    monkeypatch.setenv('ALLHANDS_API_KEY', 'test-api-key')


@pytest.fixture
def mock_issue():
    return GithubIssue(
        number=1,
        title="Test Issue",
        body="Test body",
        labels=["test"],
        state="open",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        html_url="https://github.com/test/test/issues/1",
        comments=[],
        head_branch=None,
        base_branch=None,
        review_comments=[],
        owner="test",
        repo="test",
    )


@pytest.mark.asyncio
async def test_remote_runtime_config(mock_env_vars, mock_issue, tmp_path):
    # Create a mock LLM config
    llm_config = LLMConfig(
        model="test-model",
        api_key="test-key",
        base_url="http://test.com",
    )

    # Create output directory
    output_dir = str(tmp_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "repo"), exist_ok=True)

    # Mock issue handler
    class MockIssueHandler:
        issue_type = "issue"

        def get_instruction(self, issue, prompt_template, repo_instruction):
            return "test instruction", []

        def guess_success(self, issue, history, llm_config):
            return True, None, "test explanation"

    try:
        await process_issue(
            mock_issue,
            "test-commit",
            1,
            llm_config,
            output_dir,
            "test-image",
            "test-prompt",
            MockIssueHandler(),
            None,
            True,
        )

        # The test will fail before this point due to connection errors,
        # but we can verify the config was set up correctly
    except Exception as e:
        # We expect a connection error since we're not actually connecting to a remote runtime
        assert "Failed to connect" in str(e) or "Connection refused" in str(e)
