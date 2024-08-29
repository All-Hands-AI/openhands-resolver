import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass
from github_resolver.resolve_issues import process_issue
from github_resolver.github_issue import GithubIssue
from openhands.core.config import LLMConfig
from openhands.events.observation import Observation

@dataclass
class MockObservation(Observation):
    content: str

@pytest.mark.asyncio
async def test_process_issue_adds_tests():
    # Mock dependencies
    mock_create_runtime = AsyncMock()
    mock_initialize_runtime = AsyncMock()
    mock_run_controller = AsyncMock()
    mock_complete_runtime = AsyncMock()
    mock_guess_success = MagicMock()
    mock_copytree = MagicMock()

    # Set up test data
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=27,
        title="Add tests for new functionality",
        body="We need to add tests for the new feature we implemented.",
    )
    base_commit = "abcdef1234567890"
    max_iterations = 5
    llm_config = LLMConfig(model="test_model", api_key="test_api_key")
    runtime_container_image = "test_image:latest"
    prompt_template = "Issue: {{ body }}\n\nPlease fix this issue and add appropriate tests."

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(output_dir)

        # Mock return values
        mock_create_runtime.return_value = AsyncMock()
        mock_run_controller.return_value = MagicMock(
            history=MagicMock(
                get_events=MagicMock(return_value=[
                    MockObservation(content="Created new test file: test_new_feature.py"),
                    MockObservation(content="Added test cases for the new feature"),
                ])
            ),
            metrics=MagicMock(get=MagicMock(return_value={"test_result": "passed"})),
            last_error=None,
        )
        mock_complete_runtime.return_value = {"git_patch": "test patch with new tests"}
        mock_guess_success.return_value = (True, "Issue resolved and tests added successfully")

        # Patch the necessary functions
        with patch("github_resolver.resolve_issues.create_runtime", mock_create_runtime), \
             patch("github_resolver.resolve_issues.initialize_runtime", mock_initialize_runtime), \
             patch("github_resolver.resolve_issues.run_controller", mock_run_controller), \
             patch("github_resolver.resolve_issues.complete_runtime", mock_complete_runtime), \
             patch("github_resolver.resolve_issues.guess_success", mock_guess_success), \
             patch("github_resolver.resolve_issues.logger"), \
             patch("shutil.copytree", mock_copytree):

            # Call the function
            result = await process_issue(
                issue,
                base_commit,
                max_iterations,
                llm_config,
                output_dir,
                runtime_container_image,
                prompt_template,
                reset_logger=False
            )

            # Assert the result
            assert result.success
            assert "tests added successfully" in result.success_explanation
            assert "test patch with new tests" in result.git_patch

            # Check if the history contains events related to adding tests
            history_events = mock_run_controller.return_value.history.get_events.return_value
            assert any("Created new test file" in event.content for event in history_events)
            assert any("Added test cases" in event.content for event in history_events)

            # Verify that copytree was called
            mock_copytree.assert_called_once()

if __name__ == "__main__":
    pytest.main()
