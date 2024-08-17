from typing import Any
from litellm import BaseModel
from github_resolver.github_issue import GithubIssue


class ResolverOutput(BaseModel):
    # NOTE: User-specified
    issue: GithubIssue
    instruction: str
    base_commit: str
    git_patch: str
    history: list[tuple[dict, dict]]
    metrics: dict[str, Any] | None
    error: str | None
