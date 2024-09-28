from typing import Any
from litellm import BaseModel
from openhands_resolver.github_issue import GithubIssue


class ResolverOutput(BaseModel):
    # NOTE: User-specified
    issue: GithubIssue
    instruction: str
    base_commit: str
    git_patch: str
    history: list[dict[str, Any]]
    metrics: dict[str, Any] | None
    success: bool
    success_explanation: str
    error: str | None
