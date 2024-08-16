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

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        # Apply custom serialization for metadata (to avoid leaking sensitive info)
        dumped_dict['metadata'] = json.loads(self.metadata.model_dump_json())
        return json.dumps(dumped_dict)
