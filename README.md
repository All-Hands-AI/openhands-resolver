# Github issue resolver

First, setup OpenDevin.
Add the top directory of this repo to your `PYTHONPATH`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/OpenDevin
```

Then run the following command to resolve issues in the `owner/repo` repository.
The command should be run from the OpenDevin repo

```bash
poetry run python github_resolver/main.py owner/repo
```
