from pydantic import BaseModel
from typing import Optional


class GithubIssue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    closing_issues: Optional[list[str]] = []
    review_comments: Optional[list[str]] = []
    head_branch: Optional[str] = None