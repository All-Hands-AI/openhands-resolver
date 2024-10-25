import argparse
import os
import shutil
from openhands_resolver.github_issue import GithubIssue
from openhands_resolver.io_utils import (
    load_all_resolver_outputs,
    load_single_resolver_output,
)
from openhands_resolver.patching import parse_patch, apply_diff
import requests
import subprocess
import shlex
import json

from openhands_resolver.resolver_output import ResolverOutput


def apply_patch(repo_dir: str, patch: str) -> None:
    diffs = parse_patch(patch)
    for diff in diffs:
        if not diff.header.new_path:
            print("Warning: Could not determine file to patch")
            continue

        old_path = (
            os.path.join(repo_dir, diff.header.old_path.removeprefix("b/"))
            if diff.header.old_path and diff.header.old_path != "/dev/null"
            else None
        )
        new_path = os.path.join(repo_dir, diff.header.new_path.removeprefix("b/"))

        # Check if the file is being deleted
        if diff.header.new_path == "/dev/null":
            assert old_path is not None
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Deleted file: {old_path}")
            continue

        if old_path:
            # Open the file in binary mode to detect line endings
            with open(old_path, "rb") as f:
                original_content = f.read()

            # Detect line endings
            if b"\r\n" in original_content:
                newline = "\r\n"
            elif b"\n" in original_content:
                newline = "\n"
            else:
                newline = None  # Let Python decide

            with open(old_path, "r", newline=newline) as f:
                split_content = [x.strip(newline) for x in f.readlines()]
        else:
            newline = "\n"
            split_content = []

        if diff.changes is None:
            print(f"Warning: No changes to apply for {old_path}")
            continue

        new_content = apply_diff(diff, split_content)

        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Write the new content using the detected line endings
        with open(new_path, "w", newline=newline) as f:
            for line in new_content:
                print(line, file=f)

    print("Patch applied successfully")


def initialize_repo(
    output_dir: str, issue_number: int, issue_type: str, base_commit: str | None = None
) -> str:
    src_dir = os.path.join(output_dir, "repo")
    dest_dir = os.path.join(output_dir, "patches", f"{issue_type}_{issue_number}")

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


def make_commit(repo_dir: str, issue: GithubIssue, issue_type: str) -> None:
    # Check if git username is set
    result = subprocess.run(
        f"git -C {repo_dir} config user.name",
        shell=True,
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        # If username is not set, configure git
        subprocess.run(
            f'git -C {repo_dir} config user.name "openhands" && '
            f'git -C {repo_dir} config user.email "openhands@all-hands.dev" && '
            f'git -C {repo_dir} config alias.git "git --no-pager"',
            shell=True,
            check=True,
        )
        print("Git user configured as openhands")

    result = subprocess.run(
        f"git -C {repo_dir} add .", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error adding files: {result.stderr}")
        raise RuntimeError("Failed to add files to git")

    commit_message = f"Fix {issue_type} #{issue.number}: {shlex.quote(issue.title)}"
    result = subprocess.run(
        f"git -C {repo_dir} commit -m {shlex.quote(commit_message)}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to commit changes: {result}")




def branch_exists(base_url: str, branch_name: str, headers: dict) -> bool:
    response = requests.get(f"{base_url}/branches/{branch_name}", headers=headers)
    return response.status_code == 200

def send_pull_request(
    github_issue: GithubIssue,
    github_token: str,
    github_username: str | None,
    patch_dir: str,
    pr_type: str,
    fork_owner: str | None = None,
    additional_message: str | None = None,
) -> str:
    if pr_type not in ["branch", "draft", "ready"]:
        raise ValueError(f"Invalid pr_type: {pr_type}")

    # Set up headers and base URL for GitHub API
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{github_issue.owner}/{github_issue.repo}"

    # Create a new branch with a unique name
    base_branch_name = f"openhands-fix-issue-{github_issue.number}"
    branch_name = base_branch_name
    attempt = 1

    while branch_exists(base_url, branch_name, headers):
        attempt += 1
        branch_name = f"{base_branch_name}-try{attempt}"

    # Get the default branch
    response = requests.get(f"{base_url}", headers=headers)
    response.raise_for_status()
    default_branch = response.json()["default_branch"]

    # Create and checkout the new branch
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

    username_and_token = f"{github_username}:{github_token}" if github_username else f"x-auth-token:{github_token}"
    push_command = (
        f"git -C {patch_dir} push "
        f"https://{username_and_token}@github.com/"
        f"{push_owner}/{push_repo}.git {branch_name}"
    )
    result = subprocess.run(push_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error pushing changes\n{push_command}\n{result.stderr}")
        raise RuntimeError("Failed to push changes to the remote repository")

    pr_title = f"Fix issue #{github_issue.number}: {github_issue.title}"
    pr_body = f"This pull request fixes #{github_issue.number}."
    if additional_message:
        pr_body += f"\n\n{additional_message}"
    pr_body += "\n\nAutomatic fix generated by [OpenHands](https://github.com/All-Hands-AI/OpenHands/) 🙌"
    

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

        url = pr_data["html_url"]

    print(f"{pr_type} created: {url}\n\n--- Title: {pr_title}\n\n--- Body:\n{pr_body}")

    return url

def reply_to_comment(github_token: str, comment_id: str, reply: str):
    # Opting for graphql as REST API doesn't allow reply to replies in comment threads
    query = """
            mutation($body: String!, $pullRequestReviewThreadId: ID!) {
                addPullRequestReviewThreadReply(input: { body: $body, pullRequestReviewThreadId: $pullRequestReviewThreadId }) {
                    comment {
                        id
                        body
                        createdAt
                    }
                }
            }
            """
    
    comment_reply = f"Openhands fix success summary\n\n\n{reply}"
    variables = {
        "body": comment_reply,
        "pullRequestReviewThreadId": comment_id  
    }
    url = "https://api.github.com/graphql"
    headers = {
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json"
    }

    response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
    response.raise_for_status()


def update_existing_pull_request(
    github_issue: GithubIssue,
    github_token: str,
    github_username: str | None,
    patch_dir: str,
    comment_message: str = "New openhands update",
    additional_message: str | None = None,
) -> str:
    # Set up headers and base URL for GitHub API
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{github_issue.owner}/{github_issue.repo}"   
    branch_name = github_issue.head_branch

    # Push the changes to the existing branch
    push_command = (
        f"git -C {patch_dir} push "
        f"https://{github_username}:{github_token}@github.com/"
        f"{github_issue.owner}/{github_issue.repo}.git {branch_name}"
    )

    result = subprocess.run(push_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error pushing changes: {result.stderr}")
        raise RuntimeError("Failed to push changes to the remote repository")

    pr_url = f"https://github.com/{github_issue.owner}/{github_issue.repo}/pull/{github_issue.number}"
    print(f"Updated pull request {pr_url} with new patches.")


    # Send PR Comments
    # TODO: run a summary of all comment success indicators for PR message
    if comment_message:
        comment_url = f"{base_url}/issues/{github_issue.number}/comments"
        comment_data = {
            "body": comment_message
        }
        comment_response = requests.post(comment_url, headers=headers, json=comment_data)
        if comment_response.status_code != 201:
            print(f"Failed to post comment: {comment_response.status_code} {comment_response.text}")
        else:
            print(f"Comment added to the PR: {comment_message}")
    
    if additional_message and github_issue.thread_ids:
        explanations = json.loads(additional_message)
        for count in range(len(github_issue.thread_ids)):
            reply_comment = explanations[count]
            comment_id = github_issue.thread_ids[count]
            reply_to_comment(github_token, comment_id, reply_comment)
    
    return pr_url


def process_single_issue(
    output_dir: str,
    resolver_output: ResolverOutput,
    github_token: str,
    github_username: str,
    pr_type: str,
    fork_owner: str | None,
    send_on_failure: bool,
) -> None:
    if not resolver_output.success and not send_on_failure:
        print(
            f"Issue {resolver_output.issue.number} was not successfully resolved. Skipping PR creation."
        )
        return

    issue_type = resolver_output.issue_type

    if issue_type == "issue":
        patched_repo_dir = initialize_repo(
            output_dir, 
            resolver_output.issue.number, 
            issue_type, 
            resolver_output.base_commit
        )
    elif issue_type == "pr":
        patched_repo_dir = initialize_repo(
            output_dir, 
            resolver_output.issue.number, 
            issue_type, 
            resolver_output.issue.head_branch
        )
    else:
        raise ValueError(f"Invalid issue type: {issue_type}")


    

    apply_patch(patched_repo_dir, resolver_output.git_patch)

    make_commit(patched_repo_dir, resolver_output.issue, issue_type)

    if issue_type == "pr":
        update_existing_pull_request(
            github_issue=resolver_output.issue,
            github_token=github_token,
            github_username=github_username,
            patch_dir=patched_repo_dir,
            additional_message=resolver_output.success_explanation
        )
    else:
        send_pull_request(
            github_issue=resolver_output.issue,
            github_token=github_token,
            github_username=github_username,
            patch_dir=patched_repo_dir,
            pr_type=pr_type,
            fork_owner=fork_owner,
            additional_message=resolver_output.success_explanation,
        )


def process_all_successful_issues(
    output_dir: str,
    github_token: str,
    github_username: str,
    pr_type: str,
    fork_owner: str | None,
) -> None:
    output_path = os.path.join(output_dir, "output.jsonl")
    for resolver_output in load_all_resolver_outputs(output_path):
        if resolver_output.success:
            print(f"Processing issue {resolver_output.issue.number}")
            process_single_issue(
                output_dir,
                resolver_output,
                github_token,
                github_username,
                pr_type,
                fork_owner,
                False,
            )


def main():
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
        help="Issue number to send the pull request for, or 'all_successful' to process all successful issues.",
    )
    parser.add_argument(
        "--fork-owner",
        type=str,
        default=None,
        help="Owner of the fork to push changes to (if different from the original repo owner).",
    )
    parser.add_argument(
        "--send-on-failure",
        action="store_true",
        help="Send a pull request even if the issue was not successfully resolved.",
    )
    my_args = parser.parse_args()

    github_token = (
        my_args.github_token if my_args.github_token else os.getenv("GITHUB_TOKEN")
    )
    if not github_token:
        raise ValueError(
            "Github token is not set, set via --github-token or GITHUB_TOKEN environment variable."
        )
    github_username = (
        my_args.github_username
        if my_args.github_username
        else os.getenv("GITHUB_USERNAME")
    )
    # Remove the check for github_username
    
    if not os.path.exists(my_args.output_dir):
        raise ValueError(f"Output directory {my_args.output_dir} does not exist.")

    if my_args.issue_number == "all_successful":
        process_all_successful_issues(
            my_args.output_dir,
            github_token,
            github_username,
            my_args.pr_type,
            my_args.fork_owner,
            my_args.send_on_failure,
        )
    else:
        if not my_args.issue_number.isdigit():
            raise ValueError(f"Issue number {my_args.issue_number} is not a number.")
        issue_number = int(my_args.issue_number)
        output_path = os.path.join(my_args.output_dir, "output.jsonl")
        resolver_output = load_single_resolver_output(output_path, issue_number)
        process_single_issue(
            my_args.output_dir,
            resolver_output,
            github_token,
            github_username,
            my_args.pr_type,
            my_args.fork_owner,
            my_args.send_on_failure,
        )

if __name__ == "__main__":
    main()
