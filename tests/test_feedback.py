import pytest
from unittest.mock import patch, MagicMock
from openhands_resolver.feedback import (
    FeedbackDataModel,
    store_feedback,
    get_trajectory_url,
    submit_resolver_output
)
from openhands_resolver.resolver_output import ResolverOutput
from openhands_resolver.github_issue import GithubIssue


@pytest.fixture
def mock_resolver_output():
    return ResolverOutput(
        issue=GithubIssue(
            owner="test-owner",
            repo="test-repo",
            number=123,
            title="Test Issue",
            body="Test Body",
        ),
        issue_type="issue",
        instruction="Fix the bug",
        base_commit="abc123",
        git_patch="test patch",
        history=[{"type": "test", "data": "test"}],
        metrics={"test": "test"},
        success=True,
        comment_success=[True],
        success_explanation="Fixed successfully",
        error=None,
        trajectory_url=None,
    )


def test_feedback_data_model():
    feedback = FeedbackDataModel(
        version="1.0",
        email="test@example.com",
        token="test-token",
        feedback="positive",
        permissions="private",
        trajectory=[{"type": "test", "data": "test"}]
    )
    assert feedback.version == "1.0"
    assert feedback.email == "test@example.com"
    assert feedback.token == "test-token"
    assert feedback.feedback == "positive"
    assert feedback.permissions == "private"
    assert feedback.trajectory == [{"type": "test", "data": "test"}]


def test_get_trajectory_url():
    url = get_trajectory_url("test-id")
    assert url == "https://www.all-hands.dev/share?share_id=test-id"


@patch('requests.post')
def test_store_feedback_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"feedback_id": "test-id"}'
    mock_post.return_value = mock_response

    feedback = FeedbackDataModel(
        version="1.0",
        email="test@example.com",
        token="test-token",
        feedback="positive",
        permissions="private",
        trajectory=[{"type": "test", "data": "test"}]
    )

    response = store_feedback(feedback)
    assert response == {"feedback_id": "test-id"}
    mock_post.assert_called_once()


@patch('requests.post')
def test_store_feedback_failure(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = 'Error'
    mock_post.return_value = mock_response

    feedback = FeedbackDataModel(
        version="1.0",
        email="test@example.com",
        token="test-token",
        feedback="positive",
        permissions="private",
        trajectory=[{"type": "test", "data": "test"}]
    )

    with pytest.raises(ValueError, match="Failed to store feedback: Error"):
        store_feedback(feedback)


@patch('openhands_resolver.feedback.store_feedback')
def test_submit_resolver_output_success(mock_store_feedback, mock_resolver_output):
    mock_store_feedback.return_value = {"feedback_id": "test-id"}
    
    url = submit_resolver_output(mock_resolver_output, "test-token")
    assert url == "https://www.all-hands.dev/share?share_id=test-id"
    
    mock_store_feedback.assert_called_once()
    feedback = mock_store_feedback.call_args[0][0]
    assert isinstance(feedback, FeedbackDataModel)
    assert feedback.version == "1.0"
    assert feedback.email == "openhands@all-hands.dev"
    assert feedback.token == "test-token"
    assert feedback.feedback == "positive"
    assert feedback.permissions == "private"
    assert feedback.trajectory == mock_resolver_output.history
