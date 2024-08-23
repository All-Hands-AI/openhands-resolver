import pytest

from unittest.mock import patch, MagicMock
from github_resolver.resolve_issues import (
    create_git_patch,
    initialize_runtime,
    complete_runtime,
    get_instruction,
)
from github_resolver.github_issue import GithubIssue
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation


@pytest.fixture
def mock_subprocess():
    with patch('subprocess.check_output') as mock_check_output:
        yield mock_check_output


@pytest.fixture
def mock_os():
    with patch('os.system') as mock_system, patch('os.path.join') as mock_join:
        yield mock_system, mock_join


def test_create_git_patch(mock_subprocess, mock_os):
    mock_subprocess.return_value = b'abcdef1234567890'
    mock_os[0].return_value = 0
    mock_os[1].return_value = '/path/to/workspace/123.patch'

    with patch('builtins.open', MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            'patch content'
        )

        git_id, patch_content = create_git_patch(
            '/path/to/workspace', 'main', 'fix', 123
        )

        assert git_id == 'abcdef1234567890'
        assert patch_content == 'patch content'
        mock_subprocess.assert_called_once_with(['git', 'rev-parse', 'main'])
        mock_os[0].assert_called_once_with(
            'cd /path/to/workspace && git diff main fix > 123.patch'
        )
        mock_open.assert_called_once_with('/path/to/workspace/123.patch', 'r')


async def create_cmd_output(
    exit_code: int, content: str, command_id: int, command: str
):
    return CmdOutputObservation(
        exit_code=exit_code, content=content, command_id=command_id, command=command
    )


@pytest.mark.asyncio
async def test_initialize_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(
            exit_code=0, content='', command_id=1, command='cd /workspace'
        ),
        create_cmd_output(
            exit_code=0,
            content='',
            command_id=2,
            command='git config --global core.pager ""',
        ),
    ]

    await initialize_runtime(mock_runtime)

    assert mock_runtime.run_action.call_count == 2
    mock_runtime.run_action.assert_any_call(CmdRunAction(command='cd /workspace'))
    mock_runtime.run_action.assert_any_call(
        CmdRunAction(command='git config --global core.pager ""')
    )


@pytest.mark.asyncio
async def test_complete_runtime():
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(
            exit_code=0, content='', command_id=1, command='cd /workspace'
        ),
        create_cmd_output(
            exit_code=0,
            content='',
            command_id=2,
            command='git config --global core.pager ""',
        ),
        create_cmd_output(
            exit_code=0,
            content='',
            command_id=3,
            command='git diff base_commit_hash fix',
        ),
        create_cmd_output(
            exit_code=0, content='git diff content', command_id=4, command='git apply'
        ),
    ]

    result = await complete_runtime(mock_runtime, 'base_commit_hash')

    assert result == {'git_patch': 'git diff content'}
    assert mock_runtime.run_action.call_count == 4


def test_get_instruction():
    issue = GithubIssue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue',
    )
    instruction = get_instruction(issue)

    assert (
        'Please fix the following issue for the repository in /workspace.'
        in instruction
    )
    assert 'This is a test issue' in instruction
    assert (
        'You should ONLY interact with the environment provided to you' in instruction
    )


if __name__ == '__main__':
    pytest.main()
