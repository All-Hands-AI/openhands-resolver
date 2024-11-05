import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from openhands_resolver.issue_definitions import IssueHandler
from openhands_resolver.resolve_issue import process_issue
from openhands_resolver.github_issue import GithubIssue
from openhands.events.observation import NullObservation
from openhands_resolver.resolver_output import ResolverOutput
from openhands.core.config import LLMConfig


@pytest.fixture
def mock_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = os.path.join(temp_dir, "repo")
        # Initialize a GitHub repo in "repo" and add a commit with "README.md"
        os.makedirs(repo_path)
        os.system(f"git init {repo_path}")
        readme_path = os.path.join(repo_path, "README.md")
        with open(readme_path, "w") as f:
            f.write("hello world")
        os.system(f"git -C {repo_path} add README.md")
        os.system(f"git -C {repo_path} commit -m 'Initial commit'")
        yield temp_dir


@pytest.fixture
def mock_prompt_template():
    return "Issue: {{ body }}\n\nPlease fix this issue."


@pytest.mark.asyncio
async def test_process_issue_with_list_history(mock_output_dir, mock_prompt_template):
    # Mock dependencies
    mock_create_runtime = MagicMock()
    mock_initialize_runtime = AsyncMock()
    mock_run_controller = AsyncMock()
    mock_complete_runtime = AsyncMock()
    handler_instance = MagicMock()

    # Set up test data
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
    )
    base_commit = "abcdef1234567890"
    max_iterations = 5
    llm_config = LLMConfig(model="test_model", api_key="test_api_key")
    runtime_container_image = "test_image:latest"

    # Test both cases: history as a list and history as an object with get_events()
    test_cases = [
        {
            "name": "history_as_list",
            "history": [NullObservation(content="")],
            "expected_histories": [{"content": "", "observation": "null"}]
        },
        {
            "name": "history_with_get_events",
            "history": MagicMock(get_events=lambda: [NullObservation(content="test")]),
            "expected_histories": [{"content": "test", "observation": "null"}]
        }
    ]

    for test_case in test_cases:
        # Mock return values
        mock_create_runtime.return_value = MagicMock(connect=AsyncMock())
        mock_run_controller.return_value = MagicMock(
            history=test_case["history"],
            metrics=MagicMock(get=MagicMock(return_value={"test_result": "passed"})),
            last_error=None,
        )
        mock_complete_runtime.return_value = {"git_patch": "test patch"}
        handler_instance.guess_success.return_value = (True, None, "Issue resolved successfully")
        handler_instance.get_instruction.return_value = ("Test instruction", [])
        handler_instance.issue_type = "issue"

        with patch(
            "openhands_resolver.resolve_issue.create_runtime", mock_create_runtime
        ), patch(
            "openhands_resolver.resolve_issue.initialize_runtime", mock_initialize_runtime
        ), patch(
            "openhands_resolver.resolve_issue.run_controller", mock_run_controller
        ), patch(
            "openhands_resolver.resolve_issue.complete_runtime", mock_complete_runtime
        ), patch(
            "openhands_resolver.resolve_issue.logger"
        ):
            result = await process_issue(
                issue,
                base_commit,
                max_iterations,
                llm_config,
                mock_output_dir,
                runtime_container_image,
                mock_prompt_template,
                handler_instance,
                repo_instruction=None,
                reset_logger=False
            )

            # Assert the result
            assert isinstance(result, ResolverOutput)
            assert result.issue == issue
            assert result.base_commit == base_commit
            assert result.git_patch == "test patch"
            assert result.success is True
            assert result.success_explanation == "Issue resolved successfully"
            assert result.error is None
            assert result.history == test_case["expected_histories"]