# OpenHands Github Issue Resolver 🙌

Need help resolving a GitHub issue but don't have the time to do it yourself? Let an AI agent help you out!

This tool allows you to use open-source AI agents based on [OpenHands](https://github.com/all-hands-ai/openhands)
to attempt to resolve GitHub issues automatically. While it can handle multiple issues, it's primarily designed
to help you resolve one issue at a time with high quality.

Getting started is simple - just follow the instructions below.

## Using the GitHub App

You can use OpenHands Resolver as a GitHub App that automatically attempts to fix issues and pull requests labeled with 'fix-me'. Here's how to set it up:

1. Create a new GitHub App:
   - Go to your GitHub Settings > Developer Settings > GitHub Apps > New GitHub App
   - Fill in the following details:
     - Name: Choose a name for your app (e.g., "OpenHands Resolver")
     - Homepage URL: Your app's homepage or repository URL
     - Webhook URL: The URL where your app is deployed (e.g., https://your-app.herokuapp.com/webhook)
     - Webhook Secret: Generate a secure random string
   - Permissions needed:
     - Repository:
       - Contents: Read & write
       - Issues: Read & write
       - Pull requests: Read & write
       - Metadata: Read-only
     - Subscribe to events:
       - Issues
       - Pull requests
   - Save the app and note down:
     - App ID
     - Webhook secret
     - Generate and download a private key

2. Deploy the app:
   - Deploy to your preferred platform (e.g., Heroku, Google Cloud Run, etc.)
   - Set the following environment variables:
     - Required:
       - `GITHUB_APP_ID`: Your GitHub App ID
       - `GITHUB_PRIVATE_KEY`: The contents of the private key file
       - `GITHUB_WEBHOOK_SECRET`: The webhook secret you created
       - `LLM_MODEL`: LLM model to use (e.g., "anthropic/claude-3-5-sonnet-20240620")
       - `LLM_API_KEY`: Your LLM API key
     - Optional:
       - `LLM_BASE_URL`: Base URL for LLM API (only if using a proxy)
       - `PORT`: Port for the web server (defaults to 5000)

3. Install the app:
   - Go to your GitHub App's settings
   - Click "Install App"
   - Choose which repositories to install it on

4. Usage:
   - Add the 'fix-me' label to any issue or pull request you want the AI to resolve
   - The app will:
     1. Attempt to resolve the issue/PR using OpenHands
     2. Create a draft PR if successful, or push a branch if unsuccessful
     3. Comment on the issue/PR with the results

Need help? Feel free to [open an issue](https://github.com/all-hands-ai/openhands-resolver/issues) or email us at [contact@all-hands.dev](mailto:contact@all-hands.dev).

## Manual Installation

If you prefer to run the resolver programmatically instead of using GitHub Actions, follow these steps:

1. Install the package:
```bash
pip install openhands-resolver
```

2. Create a GitHub access token:
   - Visit [GitHub's token settings](https://github.com/settings/personal-access-tokens/new)
   - Create a fine-grained token with these scopes:
     - "Content"
     - "Pull requests"
     - "Issues"
     - "Workflows"
   - If you don't have push access to the target repo, you can fork it first

3. Set up environment variables:
```bash
# GitHub credentials
export GITHUB_TOKEN="your-github-token"
export GITHUB_USERNAME="your-github-username"  # Optional, defaults to token owner

# LLM configuration
export LLM_MODEL="anthropic/claude-3-5-sonnet-20240620"  # Recommended
export LLM_API_KEY="your-llm-api-key"
export LLM_BASE_URL="your-api-url"  # Optional, for API proxies
```

Note: OpenHands works best with powerful models like Anthropic's Claude or OpenAI's GPT-4. While other models are supported, they may not perform as well for complex issue resolution.

## Resolving Issues

The resolver can automatically attempt to fix a single issue in your repository using the following command:

```bash
python -m openhands_resolver.resolve_issue --repo [OWNER]/[REPO] --issue-number [NUMBER]
```

For instance, if you want to resolve issue #100 in this repo, you would run:

```bash
python -m openhands_resolver.resolve_issue --repo all-hands-ai/openhands-resolver --issue-number 100
```

The output will be written to the `output/` directory.

If you've installed the package from source using poetry, you can use:

```bash
poetry run python openhands_resolver/resolve_issue.py --repo all-hands-ai/openhands-resolver --issue-number 100
```

For resolving multiple issues at once (e.g., in a batch process), you can use the `resolve_all_issues` command:

```bash
python -m openhands_resolver.resolve_all_issues --repo [OWNER]/[REPO] --issue-numbers [NUMBERS]
```

For example:
```bash
python -m openhands_resolver.resolve_all_issues --repo all-hands-ai/openhands-resolver --issue-numbers 100,101,102
```

## Responding to PR Comments

The resolver can also respond to comments on pull requests using:

```bash
python -m openhands_resolver.send_pull_request --issue-number PR_NUMBER --issue-type pr
```

This functionality is available both through the GitHub Actions workflow and when running the resolver locally.

## Visualizing successful PRs

To find successful PRs, you can run the following command:

```bash
grep '"success":true' output/output.jsonl | sed 's/.*\("number":[0-9]*\).*/\1/g'
```

Then you can go through and visualize the ones you'd like.

```bash
python -m openhands_resolver.visualize_resolver_output --issue-number ISSUE_NUMBER --vis-method json
```

## Uploading PRs

If you find any PRs that were successful, you can upload them.
There are three ways you can upload:

1. `branch` - upload a branch without creating a PR
2. `draft` - create a draft PR
3. `ready` - create a non-draft PR that's ready for review

```bash
python -m openhands_resolver.send_pull_request --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft
```

If you want to upload to a fork, you can do so by specifying the `fork-owner`:

```bash
python -m openhands_resolver.send_pull_request --issue-number ISSUE_NUMBER --github-username YOUR_GITHUB_USERNAME --pr-type draft --fork-owner YOUR_GITHUB_USERNAME
```

## Troubleshooting

If you have any issues, please open an issue on this github repo, we're happy to help!
Alternatively, you can [email us](mailto:contact@all-hands.dev) or join the [OpenHands Slack workspace](https://join.slack.com/t/opendevin/shared_invite/zt-2oikve2hu-UDxHeo8nsE69y6T7yFX_BA) and ask there.

