import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from github_resolver.send_pull_request import (
    apply_patch,
    load_resolver_output,
    initialize_repo,
    send_pull_request,
)
from github_resolver.resolver_output import ResolverOutput, GithubIssue


@pytest.fixture
def mock_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_github_issue():
    return GithubIssue(
        number=42,
        title="Test Issue",
        owner="test-owner",
        repo="test-repo",
        body="Test body",
    )


def test_load_resolver_output():
    mock_output_jsonl = 'tests/mock_output/output.jsonl'

    # Test loading an existing issue
    resolver_output = load_resolver_output(mock_output_jsonl, 3)
    assert isinstance(resolver_output, ResolverOutput)
    assert resolver_output.issue.number == 3
    assert resolver_output.issue.title == "Revert toggle for dark mode"
    assert resolver_output.issue.owner == "neubig"
    assert resolver_output.issue.repo == "pr-viewer"

    # Test loading a non-existent issue
    with pytest.raises(ValueError):
        load_resolver_output(mock_output_jsonl, 999)


def test_apply_patch(mock_output_dir):
    # Create a sample file in the mock repo
    sample_file = os.path.join(mock_output_dir, "sample.txt")
    with open(sample_file, "w") as f:
        f.write("Original content")

    # Create a sample patch
    patch_content = """
diff --git a/sample.txt b/sample.txt
index 9daeafb..b02def2 100644
--- a/sample.txt
+++ b/sample.txt
@@ -1 +1,2 @@
-Original content
+Updated content
+New line
"""

    # Apply the patch
    apply_patch(mock_output_dir, patch_content)

    # Check if the file was updated correctly
    with open(sample_file, "r") as f:
        updated_content = f.read()

    assert updated_content.strip() == "Updated content\nNew line".strip()


def test_initialize_repo(mock_output_dir):
    # Create some sample files in the mock repo
    os.makedirs(os.path.join(mock_output_dir, "repo"))
    with open(os.path.join(mock_output_dir, "repo", "file1.txt"), "w") as f:
        f.write("hello world")

    # Copy the repo to patches
    ISSUE_NUMBER = 3
    initialize_repo(mock_output_dir, ISSUE_NUMBER)
    patches_dir = os.path.join(mock_output_dir, "patches", f"issue_{ISSUE_NUMBER}")

    # Check if files were copied correctly
    assert os.path.exists(os.path.join(patches_dir, "file1.txt"))

    # Check file contents
    with open(os.path.join(patches_dir, "file1.txt"), "r") as f:
        assert f.read() == "hello world"


@patch('github_resolver.send_pull_request.requests.get')
@patch('github_resolver.send_pull_request.requests.post')
@patch('github_resolver.send_pull_request.os.system')
def test_send_pull_request(mock_os_system, mock_post, mock_get, mock_github_issue):
    # Mock API responses
    mock_get.side_effect = [
        MagicMock(json=lambda: {"default_branch": "main"}),
        MagicMock(json=lambda: {"object": {"sha": "test-sha"}}),
    ]
    mock_post.return_value.json.return_value = {
        "html_url": "https://github.com/test-owner/test-repo/pull/1"
    }

    # Call the function
    send_pull_request(
        github_issue=mock_github_issue,
        github_token="test-token",
        github_username="test-user",
        output_dir="/tmp/test-output",
    )

    # Assert API calls
    assert mock_get.call_count == 2
    assert mock_post.call_count == 2

    # Assert git commands
    assert mock_os_system.call_count == 2
    mock_os_system.assert_any_call("git -C /tmp/test-output checkout -b fix-issue-42")
    mock_os_system.assert_any_call(
        "git -C /tmp/test-output push "
        "https://test-user:test-token@github.com/test-owner/test-repo.git fix-issue-42"
    )
