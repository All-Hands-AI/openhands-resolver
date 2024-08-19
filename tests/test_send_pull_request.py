import os
import tempfile
import pytest

from github_resolver.send_pull_request import apply_patch, load_resolver_output, copy_repo_to_patches
from github_resolver.resolver_output import ResolverOutput


@pytest.fixture
def mock_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_load_resolver_output():
    mock_output_jsonl = 'tests/mock_output/output.jsonl'

    # Test loading an existing issue
    resolver_output = load_resolver_output(mock_output_jsonl, 3)
    assert isinstance(resolver_output, ResolverOutput)
    assert resolver_output.issue.number == 3
    assert resolver_output.issue.title == "Revert toggle for dark mode"
    assert resolver_output.issue.github_owner == "neubig"
    assert resolver_output.issue.github_repo == "pr-viewer"

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


def test_copy_repo_to_patches(mock_output_dir):
    # Create some sample files in the mock repo
    os.makedirs(os.path.join(mock_output_dir, "repo"))
    with open(os.path.join(mock_output_dir, "repo", "file1.txt"), "w") as f:
        f.write("hello world")

    # Copy the repo to patches
    ISSUE_NUMBER = 3
    copy_repo_to_patches(mock_output_dir, ISSUE_NUMBER)
    patches_dir = os.path.join(mock_output_dir, "patches", f"issue_{ISSUE_NUMBER}")

    # Check if files were copied correctly
    assert os.path.exists(os.path.join(patches_dir, "file1.txt"))

    # Check file contents
    with open(os.path.join(patches_dir, "file1.txt"), "r") as f:
        assert f.read() == "hello world"
