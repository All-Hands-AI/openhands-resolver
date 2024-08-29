import argparse
import os
import json
import shutil
from github_resolver.github_issue import GithubIssue
from github_resolver.resolver_output import ResolverOutput
import requests
import subprocess
import whatthepatch


def load_resolver_output(output_jsonl: str, issue_number: int) -> ResolverOutput:
    with open(output_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['issue']['number'] == issue_number:
                return ResolverOutput.model_validate(data)
    raise ValueError(f"Issue number {issue_number} not found in {output_jsonl}")


def apply_patch(repo_dir: str, patch: str) -> None:
    diffs = whatthepatch.parse_patch(patch)

    for diff in diffs:
        if not diff.header.new_path:
            print("Warning: Could not determine file to patch")
            continue

        old_path = (
            os.path.join(repo_dir, diff.header.old_path.lstrip('b/'))
            if diff.header.old_path and diff.header.old_path != '/dev/null'
            else None
        )
        new_path = os.path.join(repo_dir, diff.header.new_path.lstrip('b/'))

        if old_path:
            # Open the file in binary mode to detect line endings
            with open(old_path, 'rb') as f:
                original_content = f.read()

            # Detect line endings
            if b'\r\n' in original_content:
                newline = '\r\n'
            elif b'\n' in original_content:
                newline = '\n'
            else:
                newline = None  # Let Python decide

            with open(old_path, 'r', newline=newline) as f:
                split_content = [x.strip(newline) for x in f.readlines()]
        else:
            newline = '\n'
            split_content = []

        new_content = whatthepatch.apply_diff(diff, split_content)

        # Write the new content using the detected line endings
        with open(new_path, 'w', newline=newline) as f:
            for line in new_content:
                print(line, file=f)

    print("Patch applied successfully")


def initialize_repo(
    output_dir: str, issue_number: int, base_commit: str | None = None
) -> str:
    src_dir = os.path.join(output_dir, "repo")
    dest_dir = os.path.join(output_dir, "patches", f"issue_{issue_number}")

    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory {src_dir} does not exist.")

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(src_dir, dest_dir)
    print(f"Copied repository to {dest_dir}")

    if base_commit:
        result = subprocess.run(
            f"git -C {dest_dir} checkout {base_commit}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error checking out commit: {result.stderr}")
            raise RuntimeError("Failed to check out commit")

    return dest_dir


def make_commit(repo_dir: str, issue: GithubIssue) -> None:
    result = subprocess.run(
        f"git -C {repo_dir} add .", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error adding files: {result.stderr}")
        raise RuntimeError("Failed to add files to git")

    result = subprocess.run(
        f"git -C {repo_dir} commit -m 'Fix issue #{issue.number}: {issue.title}'",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error committing changes: {result.stderr}")
        raise RuntimeError("Failed to commit changes")


def send_pull_request(
    github_issue: GithubIssue,
    github_token: str,
    github_username: str,
    patch_dir: str,
    pr_type: str,
    fork_owner: str | None = None,
) -> str:

    if pr_type not in ["branch", "draft", "ready"]:
        raise ValueError(f"Invalid pr_type: {pr_type}")

    # Set up headers and base URL for GitHub API
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{github_issue.owner}/{github_issue.repo}"

    # Create a new branch
    branch_name = f"openhands-fix-issue-{github_issue.number}"

    # Get the default branch
    response = requests.get(f"{base_url}", headers=headers)
    response.raise_for_status()
    default_branch = response.json()["default_branch"]

    # Push changes to the new branch (using git command, as before)
    result = subprocess.run(
        f"git -C {patch_dir} checkout -b {branch_name}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error creating new branch: {result.stderr}")
        raise RuntimeError(
            f"Failed to create a new branch {branch_name} in {patch_dir}:"
        )

    # Determine the repository to push to (original or fork)
    push_owner = fork_owner if fork_owner else github_issue.owner
    push_repo = github_issue.repo

    push_command = (
        f"git -C {patch_dir} push "
        f"https://{github_username}:{github_token}@github.com/"
        f"{push_owner}/{push_repo}.git {branch_name}"
    )
    result = subprocess.run(push_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error pushing changes\n{push_command}\n{result.stderr}")
        raise RuntimeError("Failed to push changes to the remote repository")


    pr_title = f"Fix issue #{github_issue.number}: {github_issue.title}"
    pr_body = (
        f"This pull request fixes #{github_issue.number}."
        "\n\nAutomatic fix generated by OpenHands."
    )

    # If we are not sending a PR, we can finish early and return the
    # URL for the user to open a PR manually
    if pr_type == "branch":
        url = f"https://github.com/{push_owner}/{github_issue.repo}/compare/{branch_name}?expand=1"
    else:
        data = {
            "title": pr_title,
            "body": pr_body,
            "head": branch_name,
            "base": default_branch,
            "draft": pr_type == "draft",
        }
        response = requests.post(f"{base_url}/pulls", headers=headers, json=data)
        if response.status_code == 403:
            raise RuntimeError(
                "Failed to create pull request due to missing permissions. "
                "Make sure that the provided token has push permissions for the repository."
            )
        response.raise_for_status()
        pr_data = response.json()

        url = pr_data['html_url']

    print(f"{pr_type} created: {url}\n\n--- Title: {pr_title}\n\n--- Body:\n{pr_body}")

    return url


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Send a pull request to Github.")
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="Github token to access the repository.",
    )
    parser.add_argument(
        "--github-username",
        type=str,
        default=None,
        help="Github username to access the repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory to write the results.",
    )
    parser.add_argument(
        "--pr-type",
        type=str,
        default="draft",
        choices=["branch", "draft", "ready"],
        help="Type of the pull request to send [branch, draft, ready]",
    )
    parser.add_argument(
        "--issue-number",
        type=str,
        required=True,
        help="Issue number to send the pull request for.",
    )
    parser.add_argument(
        "--fork-owner",
        type=str,
        default=None,
        help="Owner of the fork to push changes to (if different from the original repo owner).",
    )
    my_args = parser.parse_args()

    github_token = (
        my_args.github_token if my_args.github_token else os.getenv("GITHUB_TOKEN")
    )
    if not github_token:
        raise ValueError("Github token is not set, set via --github-token or GITHUB_TOKEN environment variable.")
    github_username = (
        my_args.github_username
        if my_args.github_username
        else os.getenv("GITHUB_USERNAME")
    )
    if not github_username:
        raise ValueError("Github username is not set, set via --github-username or GITHUB_USERNAME environment variable.") 

    if not os.path.exists(my_args.output_dir):
        raise ValueError(f"Output directory {my_args.output_dir} does not exist.")

    if not my_args.issue_number.isdigit():
        raise ValueError(f"Issue number {my_args.issue_number} is not a number.")
    issue_number = int(my_args.issue_number)
    resolver_output = load_resolver_output(
        os.path.join(my_args.output_dir, "output.jsonl"),
        issue_number,
    )

    patched_repo_dir = initialize_repo(
        my_args.output_dir, resolver_output.issue.number, resolver_output.base_commit
    )

    apply_patch(patched_repo_dir, resolver_output.git_patch)

    make_commit(patched_repo_dir, resolver_output.issue)

    send_pull_request(
        github_issue=resolver_output.issue,
        github_token=github_token,
        github_username=github_username,
        patch_dir=patched_repo_dir,
        pr_type=my_args.pr_type,
        fork_owner=my_args.fork_owner,
    )
