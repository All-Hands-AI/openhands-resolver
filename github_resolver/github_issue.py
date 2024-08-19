from pydantic import BaseModel


class GithubIssue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
