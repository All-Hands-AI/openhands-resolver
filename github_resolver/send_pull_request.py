import argparse
import os
import json
import shutil
from github_resolver.resolver_output import ResolverOutput


def load_resolver_output(output_jsonl: str, issue_number: int) -> ResolverOutput:
    with open(output_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['issue']['number'] == issue_number:
                return ResolverOutput.model_validate(data)
    raise ValueError(f"Issue number {issue_number} not found in {output_jsonl}")


def apply_patch(repo_dir: str, patch: str) -> None:

    # Parse the patch content
    file_to_patch = None
    changes = []
    for line in patch.split('\n'):
        if line.startswith('+++'):
            file_to_patch = line.split()[1][2:]  # Remove 'b/' prefix
        elif line.startswith('@@ '):
            continue
        elif line.startswith('+'):
            changes.append(('add', line[1:]))
        elif line.startswith('-'):
            changes.append(('remove', line[1:]))

    if not file_to_patch:
        print("Error: Could not determine file to patch")
        return

    file_path = os.path.join(repo_dir, file_to_patch)

    new_content = []
    change_index = 0

    while change_index < len(changes):
        if changes[change_index][0] == 'remove':
            change_index += 1
        elif changes[change_index][0] == 'add':
            new_content.append(changes[change_index][1])
            change_index += 1

    with open(file_path, 'w') as f:
        f.write('\n'.join(new_content))

    print("Patch applied successfully")


def copy_repo_to_patches(output_dir: str, issue_number: int) -> str:
    src_dir = os.path.join(output_dir, "repo")
    dest_dir = os.path.join(output_dir, "patches", f"issue_{issue_number}")

    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory {src_dir} does not exist.")

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(src_dir, dest_dir)
    print(f"Copied repository to {dest_dir}")
    return dest_dir


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
        "--issue-number",
        type=int,
        default=None,
        help="Issue number to send the pull request for.",
    )
    my_args = parser.parse_args()

    github_token = (
        my_args.github_token if my_args.github_token else os.getenv("GITHUB_TOKEN")
    )
    github_username = (
        my_args.github_username
        if my_args.github_username
        else os.getenv("GITHUB_USERNAME")
    )

    if not os.path.exists(my_args.output_dir):
        raise ValueError(f"Output directory {my_args.output_dir} does not exist.")

    resolver_output = load_resolver_output(
        os.path.join(my_args.output_dir, "output.jsonl"),
        my_args.issue_number,
    )

    patched_repo_dir = copy_repo_to_patches(
        my_args.output_dir, resolver_output.issue.number
    )

    send_pull_request(
        github_token=github_token,
        github_username=github_username,
        output_dir=my_args.output_dir,
    )
