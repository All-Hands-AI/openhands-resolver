from pydantic import BaseModel


class GithubIssue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    closing_issues: list[str] | None = None
    review_comments: list[str] | None = None
    last_comment_ids: list[int] | None = None
    head_branch: str | None = None