# Github issue resolver

First, setup OpenDevin.
Add the OpenDevin repo to your `PYTHONPATH`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/OpenDevin
```

Then run the following command to resolve issues in the `owner/repo` repository.
The command should be run from the OpenDevin repo

```bash
poetry run python github_resolver/resolve_issues.py owner/repo
```
