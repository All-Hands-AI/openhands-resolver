
import os
import tempfile
import pytest
from github_resolver.send_pull_request import apply_patch

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
    patch_content = '''
diff --git a/sample.txt b/sample.txt
index 9daeafb..b02def2 100644
--- a/sample.txt
+++ b/sample.txt
@@ -1 +1,2 @@
-Original content
+Updated content
+New line
'''

    # Apply the patch
    apply_patch(mock_repo, patch_content)

    # Check if the file was updated correctly
    with open(sample_file, "r") as f:
        updated_content = f.read()

    assert updated_content.strip() == "Updated content\nNew line".strip()
