from pydantic import BaseModel


class GithubIssue(BaseModel):
    number: int
    title: str
    body: str
