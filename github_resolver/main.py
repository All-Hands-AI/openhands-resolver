# flake8: noqa: E501

import asyncio
import requests
import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import toml
from tqdm import tqdm

import agenthub
from github_resolver.github_issue import GithubIssue
from opendevin.controller.state.state import State
from opendevin.core.config import config
from opendevin.core.logger import get_console_handler
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import main
from opendevin.events.action import MessageAction
from opendevin.events.serialization.event import event_to_dict
from opendevin.runtime.docker.ssh_box import DockerSSHBox


def cleanup():
    print("Cleaning up child processes...")
    for process in mp.active_children():
        print(f"Terminating child process: {process.name}")
        process.terminate()
        process.join()


def codeact_user_response(state: State | None) -> str:
    msg = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "If you think you have modified the code in a way that fixes the issue, please run the following command: <execute_bash> exit </execute_bash>.\n"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n"
    )
    if state and state.history:
        user_msgs = [
            action
            for action, _ in state.history
            if isinstance(action, MessageAction) and action.source == "user"
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + "If you want to give up, run: <execute_bash> exit </execute_bash>.\n"
            )
    return msg


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    "CodeActAgent": codeact_user_response,
    "CodeActSWEAgent": codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    "CodeActAgent": "When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n",
    "CodeActSWEAgent": "When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n",
}


def create_git_patch(
    workspace_mount_path: str, main_branch: str, fix_branch: str, issue_number: int
) -> tuple[str, str | None]:
    """Create a git patch file between main_branch and fix_branch.

    Args:
        workspace_mount_path: Path to the workspace.
        main_branch: Main branch.
        fix_branch: Fix branch.
        issue_number: Issue number.

    Returns:
        A tuple of:
        - The original branch's git id
        - A patch to apply the fix
        or None if there is not a patch between the main and fix branch.
    """
    # Git the commit ID of the main branch
    git_id = (
        subprocess.check_output(["git", "rev-parse", main_branch])
        .decode("utf-8")
        .strip()
    )
    # Within the workspace, use git to create a patch between main_branch and fix_branch
    os.system(
        f"cd {workspace_mount_path} && git diff {main_branch} {fix_branch} > {issue_number}.patch"
    )
    git_patch_file = os.path.join(workspace_mount_path, f"{issue_number}.patch")
    with open(git_patch_file, "r") as f:
        patch_content = f.read()
    return git_id, patch_content


def process_issue(
    github_owner: str,
    github_repo: str,
    issue: GithubIssue,
    agent_class: str,
    metadata: dict,
    output_dir: str,
    reset_logger: bool = True,
) -> dict:
    """Process an issue.
    
    Args:
        github_owner: Owner of the github repo.
        github_repo: Github repository to resolve issues.
        issue: Github issue to resolve.
        agent_class: The agent class to use.
        metadata: Metadata for the run.
        output_dir: Output directory to write the results.
        reset_logger: Whether to reset the logger.
    
    Returns:
        Output of the run.    
    """

    # create issue-specific workspace dir
    # so that different agent don't interfere with each other.
    workspace_mount_path = os.path.join(
        config.workspace_mount_path, f"issue_{issue.number}"
    )
    pathlib.Path(workspace_mount_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Process-specific workspace mounted at {workspace_mount_path}")

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        # Set up logger
        log_file = os.path.join(output_dir, "infer_logs", f"issue_{issue.number}.log")
        # Remove all existing handlers from logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # add back the console handler to print ONE line
        logger.addHandler(get_console_handler())
        logger.info(
            f'Starting evaluation for issue {issue.number}.\nHint: run "tail -f {log_file}" to see live logs in a separate shell'
        )
        # Remove all existing handlers from logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    else:
        logger.info(f"Starting evaluation for issue {issue.number}.")

    # The workspace is set through global variables
    config.workspace_mount_path_in_sandbox = workspace_mount_path
    sandbox = DockerSSHBox()

    # Branch names
    fix_branch = f"fix-issue-{issue.number}"
    main_branch = "main"

    # Prepare instruction
    instruction = (
        f"Please clone the github repo https://github.com/{github_owner}/{github_repo}.git,"
        f" check out a new branch `{fix_branch}` from the `{main_branch}` branch, resolve the issue below,"
        f" and commit the changes.\n\n"
        "# Problem Statement\n"
        f"{issue.body}\n\n"
        "IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n"
        "You should NOT modify any existing test case files. If needed, you can add new test cases in a NEW file to reproduce the issue.\n"
        "You SHOULD INCLUDE PROPER INDENTATION in your edit commands.\n"
    )

    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX.get(agent_class, "")

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(
        main(
            instruction,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[agent_class],
            sandbox=sandbox,
            sid=f"{issue.number}",
        )
    )

    # Call the function in the process_issue function
    base_commit, patch_content = create_git_patch(
        workspace_mount_path, main_branch, fix_branch, issue.number
    )

    if state is None:
        raise ValueError("State should not be None.")

    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = {
        "number": issue.number,
        "instruction": instruction,
        "metadata": metadata,
        "history": [
            (event_to_dict(action), event_to_dict(obs)) for action, obs in state.history
        ],
        "git_issue": issue.model_dump(),  # SWE Bench specific
        "git_base_commit": base_commit,
        "git_patch": patch_content,
        "metrics": metrics,
        "error": state.last_error if state and state.last_error else None,
    }

    # Close the sandbox and delete the workspace
    sandbox.close()

    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = toml.load(file)
            if "selected_ids" in data:
                selected_ids = data["selected_ids"]
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f"Retained {subset.shape[0]} tasks after filtering")
                return subset
    return dataset


def download_issues_from_github(
    github_owner: str, github_repo: str, github_token: str
) -> list[GithubIssue]:
    """Download issues from Github.

    Args:
        github_owner: Owner of the github repo
        github_repo: Github repository to resolve issues.
        github_token: Github token to access the repository.

    Returns:
        List of Github issues.
    """
    url = f"https://api.github.com/repos/{github_owner}/{github_repo}/issues"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    issues = response.json()
    if not isinstance(issues, list) or any(
        [not isinstance(issue, dict) for issue in issues]
    ):
        raise ValueError("Expected list of dictionaries from Github API.")
    converted_issues = []
    for issue in issues:
        if any([issue.get(key) is None for key in ["number", "title", "body"]]):
            logger.warning(f"Skipping issue {issue} as it is missing number, title, or body.")
            continue
        converted_issues.append(
            GithubIssue(number=issue["number"], title=issue["title"], body=issue["body"])
        )
    return converted_issues


def resolve_issues(
    github_owner: str,
    github_repo: str,
    github_token: str,
    agent_class: str,
    max_iterations: int,
    limit_issues: int | None,
    num_workers: int,
    output_dir: str,
) -> None:
    """Resolve github issues.

    Args:
        github_owner: Github owner of the repo.
        github_repo: Github repository to resolve issues in form of `owner/repo`.
        github_token: Github token to access the repository.
        agent_class: The agent class to use.
        max_iterations: Maximum number of iterations to run
        limit_issues: Limit the number of issues to resolve.
        output_dir: Output directory to write the results.
    """

    # Load dataset
    issues: list[GithubIssue] = download_issues_from_github(
        github_owner, github_repo, github_token
    )
    if limit_issues is not None:
        issues = issues[:limit_issues]
        logger.info(f"Limiting resolving to first {limit_issues} issues.")

    # TEST METADATA
    assert (
        agent_class in AGENT_CLS_TO_FAKE_USER_RESPONSE_FN
    ), f"Unsupported agent class: {agent_class}"
    model_name = config.llm.model.split("/")[-1]

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir, "infer_logs")).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f"Using evaluation output directory: {output_dir}")

    metadata = {
        "agent_class": agent_class,
        "model_name": model_name,
        "max_iterations": max_iterations,
        "output_dir": output_dir,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        # get the commit id of current repo for reproducibility
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .strip(),
    }
    _agent_class = agenthub.Agent.get_cls(agent_class)
    if hasattr(_agent_class, "system_message"):
        metadata["system_message"] = _agent_class.system_message
    if hasattr(_agent_class, "in_context_example"):
        metadata["in_context_example"] = _agent_class.in_context_example
    logger.info(f"Metadata: {metadata}")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # OUTPUT FILE
    output_file = os.path.join(output_dir, "output.jsonl")
    logger.info(f"Writing evaluation output to {output_file}")
    finished_numbers = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                finished_numbers.add(data["number"])
        logger.warning(
            f"Output file {output_file} already exists. Loaded {len(finished_numbers)} finished issues."
        )
    output_fp = open(output_file, "a")

    logger.info(
        f"Evaluation started with Agent {agent_class}, model {model_name}, max iterations {max_iterations}."
    )

    # =============================================
    # filter out finished issues
    new_issues = []
    for issue in issues:
        if issue.number in finished_numbers:
            logger.info(f"Skipping issue {issue.number} as it is already finished.")
            continue
        new_issues.append(issue)
    logger.info(
        f"Finished issues: {len(finished_numbers)}, Remaining issues: {len(issues)}"
    )
    # =============================================

    pbar = tqdm(total=len(issues))

    # This function tracks the progress AND write the output to a JSONL file
    def update_progress(future):
        pbar.update(1)
        output = future.result()
        pbar.set_description(f'issue {output["number"][:10]}')
        pbar.set_postfix_str(f'Test Result: {output["test_result"]["result"]}')
        logger.info(
            f'Finished evaluation for issue {output["number"]}: {output["test_result"]["result"]}'
        )
        output_fp.write(json.dumps(output) + "\n")
        output_fp.flush()

    # This sets the multi-processing
    logger.info(f"Using {num_workers} workers for evaluation.")

    try:
        with ProcessPoolExecutor(num_workers) as executor:
            futures = []
            # This is how we perform multi-processing
            for issue in issues:
                future = executor.submit(
                    process_issue,
                    github_owner,
                    github_repo,
                    issue,
                    agent_class,
                    metadata,
                    output_dir,
                    reset_logger=bool(num_workers > 1),
                )
                future.add_done_callback(update_progress)
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Cleaning up...")
        cleanup()

    output_fp.close()
    logger.info("Evaluation finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Resolve issues from Github.")
    parser.add_argument(
        "--github-repo",
        type=str,
        required=True,
        help="Github repository to resolve issues in form of `owner/repo`.",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="Github token to access the repository.",
    )
    parser.add_argument(
        "--agent-class",
        type=str,
        default="CodeActAgent",
        help="The agent class to use.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum number of iterations to run.",
    )
    parser.add_argument(
        "--limit-issues",
        type=int,
        default=None,
        help="Limit the number of issues to resolve.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory to write the results.",
    )
    my_args = parser.parse_args()

    github_owner, github_repo = my_args.github_repo.split("/")
    github_token = (
        my_args.github_token if my_args.github_token else os.getenv("GITHUB_TOKEN")
    )
    if not github_token:
        raise ValueError("Github token is required.")

    resolve_issues(
        github_owner=github_owner,
        github_repo=github_repo,
        github_token=github_token,
        agent_class=my_args.agent_class,
        max_iterations=my_args.max_iterations,
        limit_issues=my_args.limit_issues,
        num_workers=my_args.num_workers,
        output_dir=my_args.output_dir,
    )
