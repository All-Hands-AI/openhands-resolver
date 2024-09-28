# OpenHands Github Backlog Resolver 🙌

Do you have a bunch of open github issues that need to be resolved but no
time to do it? What about asking an AI agent to do it for you instead?

This tool allows you to do just that, point towards a github repository,
and you can use open-source AI agents based on [OpenHands](https://github.com/all-hands-ai/openhands)
to attempt to resolve issues for you.

It's quite simple to get setup, just follow the instructions below.

## Setup

First, make sure that you have `poetry`
[installed](https://python-poetry.org/docs/#installing-with-the-official-installer),
then install the github resolver package:

```bash
git clone github.com/All-Hands-AI/openhands-resolver
cd openhands-resolver
poetry install
```

If you don't have one already, create a GitHub access token. You can use
[this link](https://github.com/settings/tokens/new?description=openhands-issue-resolver&scopes=repo)
to quickly generate a classic access token. Or, for additional security, you can
[create a fine-grained token](https://github.com/settings/personal-access-tokens/new)
that has "Content" and "Pull requests" scopes for the repository you
want to resolve issues in. If you don't have push access to that repo,
you can create a fork of the repo and use the fork.

Once you have your token set the `GITHUB_TOKEN` environment variable, e.g.
```bash
export GITHUB_TOKEN="your-secret-token"
```

You'll also need to have choose an `LLM_MODEL` and prepare an `LLM_API_KEY`,
for which you can follow the OpenHands setup instructions. OpenHands works
best with large, popular models like OpenAI's gpt-4o and Anthropic's Claude.

```bash
export LLM_MODEL="anthropic/claude-3-5-sonnet-20240620"
export LLM_API_KEY="sk_test_12345"
```

## Running the agent to resolve issues

Run the following command to resolve issues in the `[OWNER]/[REPO]` repository.
For instance, if you want to resolve issues in this repo, you would run:

```bash
poetry run python openhands_resolver/resolve_issues.py --repo all-hands-ai/openhands-resolver
```

The output will be written to the `output/` directory.

Alternatively, if you only want to resolve a subset of the issues, you can specify a
list of issues to resolve. For instance, if you want to resolve issues 100 and 101, you can run

```bash
poetry run python openhands_resolver/resolve_issues.py --repo all-hands-ai/openhands-resolver --issue-numbers 100,101
```

## Visualizing successful PRs

To find successful PRs, you can run the following command:

```bash
grep '"success":true' output/output.jsonl | sed 's/.*\("number":[0-9]*\).*/\1/g'
```

Then you can go through and visualize the ones you'd like.

```bash
poetry run python openhands_resolver/visualize_resolver_output.py --issue-number ISSUE_NUMBER --vis-method json
```

## Uploading PRs

If you find any PRs that were successful, you can upload them.
There are three ways you can upload

1. `branch` - upload a branch without creating a PR
2. `draft` - create a draft PR
3. `ready` - create a non-draft PR that's ready for review

```bash
poetry run python openhands_resolver/send_pull_request.py --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft
```

If you want to upload to a fork, you can do so by specifying the `fork-owner`.

```bash
poetry run python openhands_resolver/send_pull_request.py --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft --fork-owner YOUR_GITHUB_USERNAME
```

## Using the GitHub Actions Workflow

This repository includes a GitHub Actions workflow that can automatically attempt to fix issues labeled with 'fix-me'. To use this workflow in your own repository:

1. Copy the `.github/workflows/openhands-resolver.yml` file to your repository's `.github/workflows/` directory.

2. Set up the following GitHub secrets in your repository:
   - `PAT_TOKEN`: A Personal Access Token with repo scope (used for creating PRs and branches)
   - `PAT_USERNAME`: Your GitHub username
   - `LLM_MODEL`: The LLM model to use (e.g., "anthropic/claude-3-5-sonnet-20240620")
   - `LLM_API_KEY`: Your API key for the LLM service
   - `LLM_BASE_URL`: The base URL for the LLM API (if applicable)

   Note: The workflow also uses the default `GITHUB_TOKEN` secret for some operations, which is automatically provided by GitHub Actions.

3. To trigger the workflow, add the 'fix-me' label to any issue you want the AI to attempt to resolve.

The workflow will:
- Attempt to resolve the issue using the OpenHands resolver
- Create a draft PR if successful, or push a branch if unsuccessful
- Comment on the issue with the results

Note: This workflow requires the `openhands-resolver` package, which will be installed automatically during the workflow execution.


## Troubleshooting

If you have any issues, please open an issue on this github repo, we're happy
to help! Alternatively, you can join the [OpenHands Slack workspace](https://join.slack.com/t/opendevin/shared_invite/zt-2oikve2hu-UDxHeo8nsE69y6T7yFX_BA) and ask there.
