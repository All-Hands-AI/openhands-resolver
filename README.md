# Github issue resolver

This repo contains the code for the Github issue resolver agent.

## Setup

First, setup OpenHands.
Add the OpenHands repo to your `PYTHONPATH`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/OpenHands
```

If you don't have one already, create a github
[fine-grained personal access token](https://github.com/settings/tokens)
that has "Content" and "Pull requests" scopes for the repository you
want to resolve issues in.

## Running the agent to resolve issues

Then run the following command to resolve issues in the `owner/repo` repository.
The command should be run from the OpenHands repo

```bash
poetry run python github_resolver/resolve_issues.py owner/repo
```

The output will be written to `output/output.jsonl`.

## Uploading successful PRs

If you find any PRs that were successful, you can upload them.

```bash
poetry run python github_resolver/send_pull_request.py --issue-number XXX
```

## Visualizing Resolver Output

If you want to visualize the resolver output, you can use the following command.

```bash
poetry run python github_resolver/visualize_resolver_output.py --issue-number XXX --vis-method json
```
