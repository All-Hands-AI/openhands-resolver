from pydantic import BaseModel


class GithubIssue(BaseModel):
    github_owner: str
    github_repo: str
    number: int
    title: str
    body: str
