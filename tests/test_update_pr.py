
import pytest
from unittest.mock import Mock, patch
from github_resolver.update_pr import (
    update_pull_request,
    get_associated_issue_numbers,
    download_review_comments,
    process_unsolved_comments,
    make_modifications,
    check_resolved_comments,
    extract_suggestion
)

@pytest.fixture
def mock_github():
    with patch('github_resolver.update_pr.Github') as mock:
        yield mock

@pytest.fixture
def mock_repo(mock_github):
    repo = Mock()
    mock_github.return_value.get_repo.return_value = repo
    return repo

@pytest.fixture
def mock_pr(mock_repo):
    pr = Mock()
    mock_repo.get_pull.return_value = pr
    return pr

def test_update_pull_request(mock_repo, mock_pr):
    mock_pr.body = "This PR fixes #1 and #2"
    mock_pr.get_review_comments.return_value = []
    update_pull_request("test/repo", 1)
    mock_repo.get_pull.assert_called_once_with(1)

def test_get_associated_issue_numbers(mock_pr):
    mock_pr.body = "This PR fixes #1 and #2"
    issue_numbers = get_associated_issue_numbers(mock_pr)
    assert issue_numbers == [1, 2]

def test_download_review_comments(mock_pr):
    mock_comment = Mock(id=1, body="Test comment", user=Mock(login="user"), created_at="2023-01-01", path="file.py", position=1)
    mock_pr.get_review_comments.return_value = [mock_comment]
    
    comments = download_review_comments(mock_pr)
    
    assert len(comments) == 1
    assert comments[0]['id'] == 1
    assert comments[0]['body'] == "Test comment"

def test_process_unsolved_comments():
    comments = [
        {'id': 1, 'resolved': True},
        {'id': 2, 'resolved': False},
        {'id': 3, 'resolved': False}
    ]
    
    unsolved = process_unsolved_comments(comments)
    
    assert len(unsolved) == 2
    assert unsolved[0]['id'] == 2
    assert unsolved[1]['id'] == 3

def test_extract_suggestion():
    comment_body = "This is a comment\n```suggestion\nNew code\n```\nMore comment"
    suggestion = extract_suggestion(comment_body)
    assert suggestion == "New code"

def test_make_modifications(mock_repo, mock_pr):
    unsolved_comments = [{'id': 1, 'path': 'file.py', 'position': 1, 'body': "```suggestion\nNew code\n```"}]
    
    mock_file_content = Mock()
    mock_file_content.decoded_content = b"Original content\nSecond line\nThird line"
    mock_repo.get_contents.return_value = mock_file_content
    
    with patch('github_resolver.update_pr.extract_suggestion', return_value="New code"):
        make_modifications(mock_repo, mock_pr, unsolved_comments)
    
    mock_repo.get_contents.assert_called_once_with('file.py', ref=mock_pr.head.ref)
    mock_repo.update_file.assert_called_once_with(
        'file.py',
        'Address review comment: 1',
        'New code\nSecond line\nThird line',
        mock_file_content.sha,
        branch=mock_pr.head.ref
    )

def test_check_resolved_comments(mock_pr):
    unsolved_comments = [{'id': 1, 'path': 'file.py', 'position': 1, 'body': "```suggestion\nNew code\n```"}]
    mock_pr.get_files.return_value = [Mock(patch="New code")]
    
    resolved = check_resolved_comments(mock_pr, unsolved_comments)
    
    assert len(resolved) == 1
    assert resolved[0]['id'] == 1
