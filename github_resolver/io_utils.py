import json
from github_resolver.resolver_output import ResolverOutput


def load_resolver_output(output_jsonl: str, issue_number: int) -> ResolverOutput:
    with open(output_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["issue"]["number"] == issue_number:
                return ResolverOutput.model_validate(data)
    raise ValueError(f"Issue number {issue_number} not found in {output_jsonl}")
