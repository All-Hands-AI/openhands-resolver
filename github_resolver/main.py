# flake8: noqa: E501

import asyncio
from typing import Any
import requests
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


import agenthub
from github_resolver.github_issue import GithubIssue
from github_resolver.resolver_output import ResolverOutput
from opendevin.core.main import create_runtime, run_controller
from opendevin.controller.state.state import State
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import MessageAction
from opendevin.events.action import CmdRunAction
from opendevin.events.observation import CmdOutputObservation, ErrorObservation
from opendevin.core.config import (
    AppConfig,
    SandboxConfig,
)
from opendevin.core.config import LLMConfig
from opendevin.runtime.runtime import Runtime
from github_resolver.utils import (
    codeact_user_response,
    reset_logger_for_multiprocessing,
)


# Don't make this confgurable for now, unless we have other competitive agents
AGENT_CLASS = "CodeActAgent"


def cleanup():
    print("Cleaning up child processes...")
    for process in mp.active_children():
        print(f"Terminating child process: {process.name}")
        process.terminate()
        process.join()


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


async def initialize_runtime(
    runtime: Runtime,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    Currently it does nothing.
    """
    pass


async def complete_runtime(
    runtime: Runtime,
    issue: GithubIssue,  # this argument is not required, but it is used to get the workspace_dir_name
    base_commit: str,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation

    action = CmdRunAction(command=f'cd /workspace/issue_{issue.number}')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='git add -A')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {base_commit}',
            keep_prompt=False,
        )
        action.timeout = 600 + 100 * n_retries
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = await runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                await asyncio.sleep(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            await asyncio.sleep(10)
        else:
            raise ValueError(f'Unexpected observation type: {type(obs)}')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def get_instruction(
    issue: GithubIssue,
):# Prepare instruction
    instruction = (
        f'Please fix the following issue for the repository in /workspace/issue_{issue.number}.\n'
        'Environment has been set up for you to start working. You may assume all necessary tools are installed.\n\n'
        '# Problem Statement\n'
        f'{issue.body}\n\n'
        'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
        'You should NOT modify any existing test case files. If needed, you can add new test cases in a NEW file to reproduce the issue.\n'
        'You SHOULD INCLUDE PROPER INDENTATION in your edit commands.\n'
        'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.'
    )

    return instruction


async def process_issue(
    issue: GithubIssue,
    base_commit: str,
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    container_image: str | None = None,
    reset_logger: bool = True,
) -> None:

    config = AppConfig(
        default_agent="CodeActAgent",
        run_as_devin=False,
        runtime='eventstream',
        max_budget_per_task=4,
        max_iterations=max_iterations,
        sandbox=SandboxConfig(
            container_image=container_image,
            enable_auto_lint=True,
            use_host_network=False,
            # large enough timeout, since some testcases take very long to run
            timeout=300,
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(llm_config)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, issue.number, log_dir)
    else:
        logger.info(f'Starting fixing issue {issue.number}.')

    runtime = await create_runtime(config, sid=issue.number)
    await initialize_runtime(runtime)

    instruction = get_instruction(issue)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=codeact_user_response,
    )

    # Get git patch
    return_val = await complete_runtime(runtime, issue, base_commit)
    git_patch = return_val['git_patch']
    logger.info(
        f'Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------'
    )

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()
    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = ResolverOutput(
        issue=issue,
        instruction=instruction,
        git_patch=git_patch,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


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
    max_iterations: int,
    limit_issues: int | None,
    num_workers: int,
    output_dir: str,
    container_image: str | None = None,
) -> None:
    """Resolve github issues.

    Args:
        github_owner: Github owner of the repo.
        github_repo: Github repository to resolve issues in form of `owner/repo`.
        github_token: Github token to access the repository.
        max_iterations: Maximum number of iterations to run
        limit_issues: Limit the number of issues to resolve.
        output_dir: Output directory to write the results.
        container_image: Container image to use for evaluation.
    """

    # Load dataset
    issues: list[GithubIssue] = download_issues_from_github(
        github_owner, github_repo, github_token
    )
    if limit_issues is not None:
        issues = issues[:limit_issues]
        logger.info(f"Limiting resolving to first {limit_issues} issues.")

    # TEST METADATA
    model_name = llm_config.model.split("/")[-1]

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir, "infer_logs")).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f"Using evaluation output directory: {output_dir}")

    # get the commit id of current repo for reproducibility
    base_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .strip()
    )

    metadata = {
        "agent_class": AGENT_CLASS,
        "model_name": model_name,
        "max_iterations": max_iterations,
        "output_dir": output_dir,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_commit": base_commit,
    }
    _AGENT_CLASS = agenthub.Agent.get_cls(AGENT_CLASS)
    if hasattr(_AGENT_CLASS, "system_message"):
        metadata["system_message"] = _AGENT_CLASS.system_message
    if hasattr(_AGENT_CLASS, "in_context_example"):
        metadata["in_context_example"] = _AGENT_CLASS.in_context_example
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
        f"Evaluation started with Agent {AGENT_CLASS}, model {model_name}, max iterations {max_iterations}."
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
                    issue=issue,
                    base_commit=base_commit,
                    max_iterations=max_iterations,
                    llm_config=llm_config,
                    output_dir=output_dir,
                    container_image=container_image,
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
        "--container-image",
        type=str,
        default=None,
        help="Container image to use for evaluation.",
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
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use for evaluation.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key to use for evaluation.",
    )
    my_args = parser.parse_args()

    github_owner, github_repo = my_args.github_repo.split("/")
    github_token = (
        my_args.github_token if my_args.github_token else os.getenv("GITHUB_TOKEN")
    )
    if not github_token:
        raise ValueError("Github token is required.")

    llm_config = LLMConfig(
        model=my_args.llm_model or os.environ["LLM_MODEL"],
        api_key=my_args.llm_api_key or os.environ["LLM_API_KEY"],
    )

    resolve_issues(
        github_owner=github_owner,
        github_repo=github_repo,
        github_token=github_token,
        container_image=my_args.container_image,
        max_iterations=my_args.max_iterations,
        limit_issues=my_args.limit_issues,
        num_workers=my_args.num_workers,
        output_dir=my_args.output_dir,
        llm_config=llm_config,
    )
