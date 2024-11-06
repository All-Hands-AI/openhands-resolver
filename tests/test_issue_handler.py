from unittest.mock import patch, MagicMock
from openhands_resolver.issue_definitions import IssueHandler, PRHandler
from openhands_resolver.github_issue import GithubIssue
from openhands.events.action.message import MessageAction
from openhands.core.config import LLMConfig

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

def test_pr_handler_guess_success_with_thread_comments():
    # Create a PR handler instance
    handler = PRHandler('test-owner', 'test-repo', 'test-token')
    
    # Create a mock issue with thread comments but no review comments
    issue = GithubIssue(
        owner='test-owner',
        repo='test-repo',
        number=1,
        title='Test PR',
        body='Test Body',
        thread_comments=['First comment', 'Second comment'],
        closing_issues=['Issue description'],
        review_comments=None,
        thread_ids=None,
        head_branch='test-branch'
    )
    
    # Create mock history
    history = [MessageAction(content='Fixed the issue by implementing X and Y')]
    
    # Create mock LLM config
    llm_config = LLMConfig(model='test-model', api_key='test-key')
    
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""--- success
true

--- explanation
The changes successfully address the feedback."""
            )
        )
    ]
    
    # Test the guess_success method
    with patch('litellm.completion', return_value=mock_response):
        success, success_list, explanation = handler.guess_success(issue, history, llm_config)
        
        # Verify the results
        assert success is True
        assert success_list == [True]
        assert "successfully address" in explanation

def test_pr_handler_guess_success_no_comments():
    # Create a PR handler instance
    handler = PRHandler('test-owner', 'test-repo', 'test-token')
    
    # Create a mock issue with no comments
    issue = GithubIssue(
        owner='test-owner',
        repo='test-repo',
        number=1,
        title='Test PR',
        body='Test Body',
        thread_comments=None,
        closing_issues=['Issue description'],
        review_comments=None,
        thread_ids=None,
        head_branch='test-branch'
    )
    
    # Create mock history
    history = [MessageAction(content='Fixed the issue')]
    
    # Create mock LLM config
    llm_config = LLMConfig(model='test-model', api_key='test-key')
    
    # Test that it raises ValueError when no comments are present
    try:
        handler.guess_success(issue, history, llm_config)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Expected either review comments or thread comments to be initialized."
