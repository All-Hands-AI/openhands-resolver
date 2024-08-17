
import os
import tempfile
import pytest
import json
from unittest.mock import patch, MagicMock
from github_resolver.send_pull_request import apply_patch, main

@pytest.fixture
def mock_repo():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_apply_patch(mock_repo):
    # Create a sample file in the mock repo
    sample_file = os.path.join(mock_repo, "sample.txt")
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
    apply_patch(mock_repo, patch_content)

    # Check if the file was updated correctly
    with open(sample_file, "r") as f:
        updated_content = f.read()

    assert updated_content.strip() == "Updated content\nNew line".strip()

@patch('github_resolver.send_pull_request.Github')
@patch('github_resolver.send_pull_request.subprocess.run')
def test_main(mock_subprocess_run, mock_github):
    # Mock GitHub API
    mock_repo = MagicMock()
    mock_github.return_value.get_repo.return_value = mock_repo
    mock_pull_request = MagicMock()
    mock_repo.create_pull.return_value = mock_pull_request

    # Mock subprocess.run
    mock_subprocess_run.return_value.returncode = 0

    # Prepare mock input data
    with open('tests/mock_output/mock_output.json', 'r') as f:
        mock_data = json.load(f)

    # Prepare temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample file in the temp directory
        sample_file = os.path.join(temp_dir, "sample.txt")
        with open(sample_file, "w") as f:
            f.write("Original content")

        # Run the main function
        main(repo_name="test/repo", branch_name="test-branch", commit_message="Test commit", 
             title="Test PR", body=mock_data['pull_request_body'], patch=mock_data['patch'])

        # Assert that the patch was applied
        with open(sample_file, "r") as f:
            updated_content = f.read()
        assert "Updated content" in updated_content

        # Assert that GitHub API calls were made
        mock_github.assert_called_once()
        mock_repo.create_pull.assert_called_once_with(
            title="Test PR",
            body=mock_data['pull_request_body'],
            head="test-branch",
            base="main"
        )

        # Assert that git commands were executed
        assert mock_subprocess_run.call_count == 3  # git add, git commit, git push
