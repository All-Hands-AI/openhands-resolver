import pytest

def test_required_imports():
    """Test that all required imports are available."""
    try:
        from openhands.events.stream import EventStream
    except ImportError as e:
        pytest.fail(f"Failed to import EventStream: {str(e)}")

    try:
        from openhands_resolver.issue_definitions import (
            IssueHandler,
            PRHandler,
            IssueHandlerInterface
        )
    except ImportError as e:
        pytest.fail(f"Failed to import from issue_definitions: {str(e)}")