# Github issue resolver

First, setup OpenHands.
Add the OpenHands repo to your `PYTHONPATH`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/OpenHands
```

If you don't have one already, create a github
[fine-grained personal access token](https://github.com/settings/tokens)
that has "Content" and "Pull requests" scopes for the repository you
want to resolve issues in.

Then run the following command to resolve issues in the `owner/repo` repository.
The command should be run from the OpenHands repo

```bash
poetry run python github_resolver/resolve_issues.py owner/repo
```
