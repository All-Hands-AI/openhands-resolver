import os
import tempfile
import pytest


from unittest.mock import AsyncMock, patch, MagicMock
from github_resolver.resolve_issues import (
    create_git_patch,
    initialize_runtime,
    complete_runtime,
    get_instruction,
    process_issue,
    download_issues_from_github,
)
from github_resolver.github_issue import GithubIssue
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, NullObservation
from github_resolver.resolver_output import ResolverOutput
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
def mock_subprocess():
    with patch("subprocess.check_output") as mock_check_output:
        yield mock_check_output


@pytest.fixture
def mock_os():
    with patch("os.system") as mock_system, patch("os.path.join") as mock_join:
        yield mock_system, mock_join


@pytest.fixture
def mock_prompt_template():
    return "Issue: {{ body }}\n\nPlease fix this issue."


def test_create_git_patch(mock_subprocess, mock_os):
    mock_subprocess.return_value = b"abcdef1234567890"
    mock_os[0].return_value = 0
    mock_os[1].return_value = "/path/to/workspace/123.patch"

    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "patch content"
        )

        git_id, patch_content = create_git_patch(
            "/path/to/workspace", "main", "fix", 123
        )

        assert git_id == "abcdef1234567890"
        assert patch_content == "patch content"
        mock_subprocess.assert_called_once_with(["git", "rev-parse", "main"])
        mock_os[0].assert_called_once_with(
            "cd /path/to/workspace && git diff main fix > 123.patch"
        )
        mock_open.assert_called_once_with("/path/to/workspace/123.patch", "r")


def create_cmd_output(
    exit_code: int, content: str, command_id: int, command: str
):
    return CmdOutputObservation(
        exit_code=exit_code, content=content, command_id=command_id, command=command
    )


def test_initialize_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(
            exit_code=0, content="", command_id=1, command="cd /workspace"
        ),
        create_cmd_output(
            exit_code=0,
            content="",
            command_id=2,
            command='git config --global core.pager ""',
        ),
    ]

    initialize_runtime(mock_runtime)

    assert mock_runtime.run_action.call_count == 2
    mock_runtime.run_action.assert_any_call(CmdRunAction(command="cd /workspace"))
    mock_runtime.run_action.assert_any_call(
        CmdRunAction(command='git config --global core.pager ""')
    )


def test_download_issues_from_github():
    mock_response = MagicMock()
    mock_response.json.side_effect = [
        [
            {"number": 1, "title": "Issue 1", "body": "This is an issue"},
            {"number": 2, "title": "PR 1", "body": "This is a pull request", "pull_request": {}},
            {"number": 3, "title": "Issue 2", "body": "This is another issue"},
        ],
        None,
    ]
    mock_response.raise_for_status = MagicMock()

    with patch('requests.get', return_value=mock_response):
        issues = download_issues_from_github("owner", "repo", "token")

    assert len(issues) == 2
    assert all(isinstance(issue, GithubIssue) for issue in issues)
    assert [issue.number for issue in issues] == [1, 3]
    assert [issue.title for issue in issues] == ["Issue 1", "Issue 2"]

@pytest.mark.asyncio
async def test_complete_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(
            exit_code=0, content="", command_id=1, command="cd /workspace"
        ),
        create_cmd_output(
            exit_code=0,
            content="",
            command_id=2,
            command='git config --global core.pager ""',
        ),
        create_cmd_output(
            exit_code=0,
            content="",
            command_id=3,
            command='git config --global --add safe.directory /workspace',
        ),
        create_cmd_output(
            exit_code=0,
            content="",
            command_id=4,
            command="git diff base_commit_hash fix",
        ),
        create_cmd_output(
            exit_code=0, content="git diff content", command_id=5, command="git apply"
        ),
    ]

    result = await complete_runtime(mock_runtime, "base_commit_hash")

    assert result == {"git_patch": "git diff content"}
    assert mock_runtime.run_action.call_count == 5


@pytest.mark.asyncio
async def test_process_issue(mock_output_dir, mock_prompt_template):
    # Mock dependencies
    mock_create_runtime = AsyncMock()
    mock_initialize_runtime = AsyncMock()
    mock_run_controller = AsyncMock()
    mock_complete_runtime = AsyncMock()
    mock_guess_success = MagicMock()

    # Set up test data
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=1,
        title="Test Issue",
        body="This is a test issue",
    )
    base_commit = "abcdef1234567890"
    repo_instruction = "Resolve this repo"
    max_iterations = 5
    llm_config = LLMConfig(model="test_model", api_key="test_api_key")
    runtime_container_image = "test_image:latest"

    # Mock return values
    mock_create_runtime.return_value = MagicMock()
    mock_run_controller.return_value = MagicMock(
        history=MagicMock(
            get_events=MagicMock(return_value=[NullObservation(content="")])
        ),
        metrics=MagicMock(get=MagicMock(return_value={"test_result": "passed"})),
        last_error=None,
    )
    mock_complete_runtime.return_value = {"git_patch": "test patch"}
    mock_guess_success.return_value = (True, "Issue resolved successfully")

    # Patch the necessary functions
    with patch(
        "github_resolver.resolve_issues.create_runtime", mock_create_runtime
    ), patch(
        "github_resolver.resolve_issues.initialize_runtime", mock_initialize_runtime
    ), patch(
        "github_resolver.resolve_issues.run_controller", mock_run_controller
    ), patch(
        "github_resolver.resolve_issues.complete_runtime", mock_complete_runtime
    ), patch(
        "github_resolver.resolve_issues.guess_success", mock_guess_success
    ), patch(
        "github_resolver.resolve_issues.logger"
    ):

        # Call the function
        result = await process_issue(
            issue,
            base_commit,
            max_iterations,
            llm_config,
            mock_output_dir,
            runtime_container_image,
            mock_prompt_template,  # Add this argument
            repo_instruction,
            reset_logger=False
        )

        # Assert the result
        assert isinstance(result, ResolverOutput)
        assert result.issue == issue
        assert result.base_commit == base_commit
        assert result.git_patch == "test patch"
        assert result.success
        assert result.success_explanation == "Issue resolved successfully"
        assert result.error is None

        # Assert that the mocked functions were called
        mock_create_runtime.assert_called_once()
        mock_initialize_runtime.assert_called_once()
        mock_run_controller.assert_called_once()
        mock_complete_runtime.assert_called_once()
        mock_guess_success.assert_called_once()


def test_get_instruction(mock_prompt_template):
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=123,
        title="Test Issue",
        body="This is a test issue",
    )
    instruction = get_instruction(issue, mock_prompt_template, None)
    expected_instruction = "Issue: This is a test issue\n\nPlease fix this issue."
    assert instruction == expected_instruction


def test_file_instruction():
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=123,
        title="Test Issue",
        body="This is a test issue",
    )
    # load prompt from github_resolver/prompts/resolve/basic.jinja
    with open("github_resolver/prompts/resolve/basic.jinja", "r") as f:
        prompt = f.read()
    instruction = get_instruction(issue, prompt, None)
    expected_instruction = """Please fix the following issue for the repository in /workspace.
An environment has been set up for you to start working. You may assume all necessary tools are installed.

# Problem Statement
This is a test issue

IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
You SHOULD INCLUDE PROPER INDENTATION in your edit commands.

When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>."""
    assert instruction == expected_instruction


def test_file_instruction_with_repo_instruction():
    issue = GithubIssue(
        owner="test_owner",
        repo="test_repo",
        number=123,
        title="Test Issue",
        body="This is a test issue",
    )
    # load prompt from github_resolver/prompts/resolve/basic.jinja
    with open("github_resolver/prompts/resolve/basic.jinja", "r") as f:
        prompt = f.read()
    # load repo instruction from github_resolver/prompts/repo_instructions/all-hands-ai___github-resolver.txt
    with open("github_resolver/prompts/repo_instructions/all-hands-ai___github-resolver.txt", "r") as f:
        repo_instruction = f.read()
    instruction = get_instruction(issue, prompt, repo_instruction)
    expected_instruction = """Please fix the following issue for the repository in /workspace.
An environment has been set up for you to start working. You may assume all necessary tools are installed.

# Problem Statement
This is a test issue

IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
You SHOULD INCLUDE PROPER INDENTATION in your edit commands.

Some basic information about this repository:
This is a Python repo based on the poetry package manager.
- Setup: `poetry install --with test --with dev`
- Testing: `poetry run pytest tests/test_*.py`

When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>."""
    assert instruction == expected_instruction


if __name__ == "__main__":
    pytest.main()
