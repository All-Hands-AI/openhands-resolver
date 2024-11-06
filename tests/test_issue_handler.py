from unittest.mock import patch, MagicMock
from openhands_resolver.issue_definitions import IssueHandler

def test_get_converted_issues_initializes_review_comments():
    # Mock the necessary dependencies
    with patch('requests.get') as mock_get:
        # Mock the response for issues
        mock_issues_response = MagicMock()
        mock_issues_response.json.return_value = [{
            'number': 1,
            'title': 'Test Issue',
            'body': 'Test Body'
        }]
        # Mock the response for comments
        mock_comments_response = MagicMock()
        mock_comments_response.json.return_value = []
        
        # Set up the mock to return different responses for different calls
        # First call is for issues, second call is for comments
        mock_get.side_effect = [mock_issues_response, mock_comments_response, mock_comments_response]  # Need two comment responses because we make two API calls
        
        # Create an instance of IssueHandler
        handler = IssueHandler('test-owner', 'test-repo', 'test-token')
        
        # Get converted issues
        issues = handler.get_converted_issues()
        
        # Verify that we got exactly one issue
        assert len(issues) == 1
        
        # Verify that review_comments is initialized as None
        assert issues[0].review_comments is None
        
        # Verify other fields are set correctly
        assert issues[0].number == 1
        assert issues[0].title == 'Test Issue'
        assert issues[0].body == 'Test Body'
        assert issues[0].owner == 'test-owner'
        assert issues[0].repo == 'test-repo'
