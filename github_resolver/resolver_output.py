import json
from typing import Any
from litellm import BaseModel
from github_resolver.github_issue import GithubIssue


class ResolverOutput(BaseModel):
    # NOTE: User-specified
    issue: GithubIssue
    instruction: str
    git_patch: str
    history: list[tuple[dict, dict]]
    metrics: dict[str, Any] | None
    error: str | None
