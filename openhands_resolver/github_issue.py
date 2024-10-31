from pydantic import BaseModel
from typing import TypedDict


class ReviewComment(TypedDict):
    comment: str
    files: list[str]


class GithubIssue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    thread_comments: list[str] | None = None  # Added field for issue thread comments
    closing_issues: list[str] | None = None
    review_comments: list[ReviewComment] | None = None
    thread_ids: list[str] | None = None
    head_branch: str | None = None

