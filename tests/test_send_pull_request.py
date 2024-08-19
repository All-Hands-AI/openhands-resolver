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


def test_apply_patch_preserves_line_endings(mock_output_dir):
    # Create sample files with different line endings
    unix_file = os.path.join(mock_output_dir, "unix_style.txt")
    dos_file = os.path.join(mock_output_dir, "dos_style.txt")

    with open(unix_file, "w", newline='\n') as f:
        f.write("Line 1\nLine 2\nLine 3")

    with open(dos_file, "w", newline='\r\n') as f:
        f.write("Line 1\r\nLine 2\r\nLine 3")

    # Create patches for both files
    unix_patch = """
diff --git a/unix_style.txt b/unix_style.txt
index 9daeafb..b02def2 100644
--- a/unix_style.txt
+++ b/unix_style.txt
@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Updated Line 2
 Line 3
"""

    dos_patch = """
diff --git a/dos_style.txt b/dos_style.txt
index 9daeafb..b02def2 100644
--- a/dos_style.txt
+++ b/dos_style.txt
@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Updated Line 2
 Line 3
"""

    # Apply patches
    apply_patch(mock_output_dir, unix_patch)
    apply_patch(mock_output_dir, dos_patch)

    # Check if line endings are preserved
    with open(unix_file, "rb") as f:
        unix_content = f.read()
    with open(dos_file, "rb") as f:
        dos_content = f.read()

    assert b'\r\n' not in unix_content, "Unix-style line endings were changed to DOS-style"
    assert b'\r\n' in dos_content, "DOS-style line endings were changed to Unix-style"

    # Check if content was updated correctly
    assert unix_content.decode('utf-8').split('\n')[1] == "Updated Line 2"
    assert dos_content.decode('utf-8').split('\r\n')[1] == "Updated Line 2"


def test_apply_patch_with_dev_null(mock_output_dir):
    # Create a patch that adds a new file
    patch_content = """
diff --git a/new_file.txt b/new_file.txt
new file mode 100644
index 0000000..3b18e51
--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1 @@
+hello world
"""

    # Apply the patch
    apply_patch(mock_output_dir, patch_content)

    # Check if the new file was created
    new_file_path = os.path.join(mock_output_dir, "new_file.txt")
    assert os.path.exists(new_file_path), "New file was not created"

    # Check if the file content is correct
    with open(new_file_path, "r") as f:
        content = f.read().strip()
    assert content == "hello world", "File content is incorrect"


def test_initialize_repo(mock_output_dir):
    # Copy the repo to patches
    ISSUE_NUMBER = 3
    initialize_repo(mock_output_dir, ISSUE_NUMBER)
    patches_dir = os.path.join(mock_output_dir, "patches", f"issue_{ISSUE_NUMBER}")

    # Check if files were copied correctly
    assert os.path.exists(os.path.join(patches_dir, "README.md"))

    # Check file contents
    with open(os.path.join(patches_dir, "README.md"), "r") as f:
        assert f.read() == "hello world"


@patch('subprocess.run')
@patch('requests.post')
@patch('requests.get')
def test_send_pull_request(
    mock_get, mock_post, mock_run, mock_github_issue, mock_output_dir
):
    repo_path = os.path.join(mock_output_dir, "repo")

    # Mock API responses
    mock_get.side_effect = [
        MagicMock(json=lambda: {"default_branch": "main"}),
        MagicMock(json=lambda: {"object": {"sha": "test-sha"}}),
    ]
    mock_post.return_value.json.return_value = {
        "html_url": "https://github.com/test-owner/test-repo/pull/1"
    }

    # Mock subprocess.run calls
    mock_run.side_effect = [
        MagicMock(returncode=0),  # git checkout -b
        MagicMock(returncode=0),  # git push
    ]

    # Call the function
    result = send_pull_request(
        github_issue=mock_github_issue,
        github_token="test-token",
        github_username="test-user",
        patch_dir=repo_path,
    )

    # Assert API calls
    assert mock_get.call_count == 1
    assert mock_post.call_count == 1

    # Assert subprocess.run calls
    assert mock_run.call_count == 2

    # Check the git checkout -b command
    checkout_call = mock_run.call_args_list[0]
    assert checkout_call[0][0].startswith(
        f"git -C {repo_path} checkout -b fix-issue-42"
    )
    assert checkout_call[1] == {'shell': True, 'capture_output': True, 'text': True}

    # Check the git push command
    push_call = mock_run.call_args_list[1]
    assert push_call[0][0].startswith(
        f"git -C {repo_path} push https://test-user:test-token@github.com/test-owner/test-repo.git fix-issue-42"
    )
    assert push_call[1] == {'shell': True, 'capture_output': True, 'text': True}

    # Assert the result
    assert result == "https://github.com/test-owner/test-repo/pull/1"


@patch('subprocess.run')
@patch('requests.post')
@patch('requests.get')
def test_send_pull_request_git_push_failure(
    mock_get, mock_post, mock_run, mock_github_issue, mock_output_dir
):

    repo_path = os.path.join(mock_output_dir, "repo")

    # Mock API responses
    mock_get.return_value = MagicMock(json=lambda: {"default_branch": "main"})

    # Mock the subprocess.run calls
    mock_run.side_effect = [
        MagicMock(returncode=0),  # git checkout -b
        MagicMock(returncode=1, stderr="Error: failed to push some refs"),  # git push
    ]

    # Test that RuntimeError is raised when git push fails
    with pytest.raises(
        RuntimeError, match="Failed to push changes to the remote repository"
    ):
        send_pull_request(
            github_issue=mock_github_issue,
            github_token="test-token",
            github_username="test-user",
            patch_dir=repo_path,
        )

    # Assert that subprocess.run was called twice
    assert mock_run.call_count == 2

    # Check the git checkout -b command
    checkout_call = mock_run.call_args_list[0]
    assert checkout_call[0][0].startswith(f"git -C {repo_path} checkout -b")
    assert checkout_call[1] == {'shell': True, 'capture_output': True, 'text': True}

    # Check the git push command
    push_call = mock_run.call_args_list[1]
    assert push_call[0][0].startswith(
        f"git -C {repo_path} push https://test-user:test-token@github.com/"
    )
    assert push_call[1] == {'shell': True, 'capture_output': True, 'text': True}

    # Assert that no pull request was created
    mock_post.assert_not_called()