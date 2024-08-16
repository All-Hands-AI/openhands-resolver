import json
from litellm import BaseModel
from github_resolver.github_issue import GithubIssue


class ResolverOutput(BaseModel):
    # NOTE: User-specified
    issue: GithubIssue
    instruction: str

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        # Apply custom serialization for metadata (to avoid leaking sensitive info)
        dumped_dict['metadata'] = json.loads(self.metadata.model_dump_json())
        return json.dumps(dumped_dict)
