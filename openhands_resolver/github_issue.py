from pydantic import BaseModel


class GithubIssue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    thread_comments: list[str] | None = None  # Added field for issue thread comments
    closing_issues: list[str] | None = None
    review_comments: list[str] | None = None
    thread_ids: list[str] | None = None
    head_branch: str | None = None

