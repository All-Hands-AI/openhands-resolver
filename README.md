# OpenHands Github Backlog Resolver 🙌

Do you have a bunch of open github issues that need to be resolved but no
time to do it? What about asking an AI agent to do it for you instead?

This tool allows you to do just that, point towards a github repository,
and you can use open-source AI agents based on [OpenHands](https://github.com/all-hands-ai/openhands)
to attempt to resolve issues for you.

It's quite simple to get setup, just follow the instructions below.

## Using the GitHub Actions Workflow

This repository includes a GitHub Actions workflow that can automatically attempt to fix issues labeled with 'fix-me'.
Follow the steps to use this workflow in your own repository, and feel free to contact us through github issues or [contact@all-hands.dev](mailto:contact@all-hands.dev) if you have questions:

1. Prepare a github personal access token. You can:
    1. [Contact us](mailto:contact@all-hands.dev) and we will set up a token for the [openhands-agent](https://github.com/openhands-agent) account (if you want to make it clear which commits came from the agent.
    2. Choose your own github user that will make the commits to the repo, [and create a personal access token](https://github.com/settings/tokens?type=beta) with read/write scope for "contents", "issues", "pull requests", and "workflows" on the desired repos.

2. Create an API key for the [Claude API](https://www.anthropic.com/api) (you can also use GPT, but Claude has better performance).

3. Copy the `examples/openhands-resolver.yml` file to your repository's `.github/workflows/` directory.

4. Enable read/write workflows for the repository by going to `Settings -> Actions -> General -> Workflow permissions` and selecting "Read and write permissions" and click "Allow Github Actions to create and approve pull requests".

5. Set up the following [GitHub secrets](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions) in your repository, or across your entire org if you want to only set ths once and use the resolver in multiple repositories:
   - `PAT_USERNAME`: The github username that you used to create the personal access token.
   - `PAT_TOKEN`: The personal access token for github.
   - `LLM_MODEL`: The LLM model to use (e.g., "anthropic/claude-3-5-sonnet-20240620")
   - `LLM_API_KEY`: Your API key for the LLM service
   - `LLM_BASE_URL`: The base URL for the LLM API (optional, only if using a proxy)
In addition, if you want to send pull requests to a repo that's not your own, you can create a fork that you own, and set the following secret:
   - `PAT_FORK_OWNER`: The username of the fork owner. This is usally the same `PAT_USERNAME`.

6. To trigger the workflow, add the 'fix-me' label to any issue you want the AI to attempt to resolve.

The workflow will:

- Attempt to resolve the issue using the OpenHands resolver
- Create a draft PR if successful, or push a branch if unsuccessful
- Comment on the issue with the results

## Installation

If you want to instead run the resolver on your own programmatically.

```bash
pip install openhands-resolver
```

If you don't have one already, create a GitHub access token. You can
[create a fine-grained token](https://github.com/settings/personal-access-tokens/new)
that has "Content", "Pull requests", "Issues", and "Workflows" scopes for the repository you
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

After installing the package, you can run the agent to resolve issues using the following command:

```bash
python -m openhands_resolver.resolve_issues --repo [OWNER]/[REPO]
```

For instance, if you want to resolve issues in this repo, you would run:

```bash
python -m openhands_resolver.resolve_issues --repo all-hands-ai/openhands-resolver
```

The output will be written to the `output/` directory.

Alternatively, if you only want to resolve a subset of the issues, you can specify a
list of issues to resolve. For instance, if you want to resolve issues 100 and 101, you can run:

```bash
python -m openhands_resolver.resolve_issues --repo all-hands-ai/openhands-resolver --issue-numbers 100,101
```

If you've installed the package from source using poetry, you can still use the previous method:

```bash
poetry run python openhands_resolver/resolve_issues.py --repo all-hands-ai/openhands-resolver
```

## Visualizing successful PRs

To find successful PRs, you can run the following command:

```bash
grep '"success":true' output/output.jsonl | sed 's/.*\("number":[0-9]*\).*//g'
```

Then you can go through and visualize the ones you'd like.

```bash
python -m openhands_resolver.visualize_resolver_output --issue-number ISSUE_NUMBER --vis-method json
```

## Uploading PRs

If you find any PRs that were successful, you can upload them.
There are three ways you can upload

1. `branch` - upload a branch without creating a PR
2. `draft` - create a draft PR
3. `ready` - create a non-draft PR that's ready for review

```bash
python -m openhands_resolver.send_pull_request --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft
```

If you want to upload to a fork, you can do so by specifying the `fork-owner`.

```bash
python -m openhands_resolver.send_pull_request --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft --fork-owner YOUR_GITHUB_USERNAME
```

## Troubleshooting

If you have any issues, please open an issue on this github repo, we're happy to help!
Alternatively, you can [email us](mailto:contact@all-hands.dev) or join the [OpenHands Slack workspace](https://join.slack.com/t/opendevin/shared_invite/zt-2oikve2hu-UDxHeo8nsE69y6T7yFX_BA) and ask there.
