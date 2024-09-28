# flake8: noqa: E501

import asyncio
import dataclasses
import re
import shutil
from typing import Any, Awaitable, TextIO
import requests
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import jinja2

from tqdm import tqdm

from litellm import completion

from openhands_resolver.github_issue import GithubIssue
from openhands_resolver.resolver_output import ResolverOutput
import openhands
from openhands.core.main import create_runtime, run_controller
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.memory.history import ShortTermHistory
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
)
from openhands.core.config import LLMConfig
from openhands.runtime.runtime import Runtime
from openhands_resolver.utils import (
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


def initialize_runtime(
    runtime: Runtime,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    Currently it does nothing.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(
            f"Failed to change directory to /workspace.\n{obs}"
        )

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config.\n{obs}")


async def complete_runtime(
    runtime: Runtime,
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
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(
            f"Failed to change directory to /workspace. Observation: {obs}"
        )

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config. Observation: {obs}")

    action = CmdRunAction(command='git config --global --add safe.directory /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config. Observation: {obs}")

    action = CmdRunAction(command='git add -A')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to git add. Observation: {obs}")

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {base_commit}',
            keep_prompt=False,
        )
        action.timeout = 600 + 100 * n_retries
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
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
    prompt_template: str,
    repo_instruction: str | None = None,
):  # Prepare instruction
    template = jinja2.Template(prompt_template)
    instruction = template.render(body=issue.body, repo_instruction=repo_instruction)
    return instruction


def guess_success(issue: GithubIssue, history: ShortTermHistory, llm_config: LLMConfig) -> tuple[bool, str]:
    """Guess if the issue is fixed based on the history and the issue description."""
    
    last_message = history.get_events_as_list()[-1].message
    
    prompt = f"""Given the following issue description and the last message from an AI agent attempting to fix it, determine if the issue has been successfully resolved.

Issue description:
{issue.body}

Last message from AI agent:
{last_message}

(1) has the issue been successfully resolved?
(2) If the issue has been resolved, please provide an explanation of what was done in the PR that can be sent to a human reviewer on github. If the issue has not been resolved, please provide an explanation of why.

Answer in JSON in the format below:

```json
{{
    "success": true/false,
    "explanation": "..."
}}
```
"""

    response = completion(
        model=llm_config.model,
        messages=[{"role": "user", "content": prompt}],
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )
    
    answer = response.choices[0].message.content.strip()
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, answer)
    try:
        if match:
            json_answer = json.loads(match.group(1))
        else:
            json_answer = json.loads(answer)
        return json_answer['success'], json_answer['explanation']
    except json.JSONDecodeError:
        return False, f"Failed to decode JSON from LLM response: {answer}"


async def process_issue(
    issue: GithubIssue,
    base_commit: str,
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    runtime_container_image: str,
    prompt_template: str,
    repo_instruction: str | None = None,
    reset_logger: bool = True,
) -> ResolverOutput:

    # Setup the logger properly, so you can run multi-processing to parallelize processing
    if reset_logger:
        log_dir = os.path.join(output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
    else:
        logger.info(f'Starting fixing issue {issue.number}.')

    workspace_base = os.path.join(output_dir, "workspace", f"issue_{issue.number}")
    # Get the absolute path of the workspace base
    workspace_base = os.path.abspath(workspace_base)
    # write the repo to the workspace
    if os.path.exists(workspace_base):
        shutil.rmtree(workspace_base)
    shutil.copytree(os.path.join(output_dir, "repo"), workspace_base)

    config = AppConfig(
        default_agent="CodeActAgent",
        runtime='eventstream',
        max_budget_per_task=4,
        max_iterations=max_iterations,
        sandbox=SandboxConfig(
            runtime_container_image=runtime_container_image,
            enable_auto_lint=False,
            use_host_network=False,
            # large enough timeout, since some testcases take very long to run
            timeout=300,
        ),
        # do not mount workspace
        workspace_base=workspace_base,
        workspace_mount_path=workspace_base,
    )
    config.set_llm_config(llm_config)

    runtime = create_runtime(config, sid=f"{issue.number}")
    initialize_runtime(runtime)

    instruction = get_instruction(issue, prompt_template, repo_instruction)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=codeact_user_response,
    )
    if state is None:
        raise RuntimeError("Failed to run the agent.")

    # Get git patch
    return_val = await complete_runtime(runtime, base_commit)
    git_patch = return_val['git_patch']
    logger.info(
        f'Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------'
    )

    # Serialize histories
    histories = [dataclasses.asdict(event) for event in state.history.get_events()]
    metrics = state.metrics.get() if state.metrics else None

    # determine success based on the history and the issue description
    success, success_explanation = guess_success(issue, state.history, llm_config)

    # Save the output
    output = ResolverOutput(
        issue=issue,
        instruction=instruction,
        base_commit=base_commit,
        git_patch=git_patch,
        history=histories,
        metrics=metrics,
        success=success,
        success_explanation=success_explanation,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def download_issues_from_github(
    owner: str, repo: str, token: str
) -> list[GithubIssue]:
    """Download issues from Github.

    Args:
        owner: Owner of the github repo
        repo: Github repository to resolve issues.
        token: Github token to access the repository.

    Returns:
        List of Github issues.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params: dict[str, int | str] = {"state": "open", "per_page": 100, "page": 1}
    all_issues = []

    while True:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues = response.json()

        if not issues:
            break

        if not isinstance(issues, list) or any(
            [not isinstance(issue, dict) for issue in issues]
        ):
            raise ValueError("Expected list of dictionaries from Github API.")

        all_issues.extend(issues)
        assert isinstance(params["page"], int)
        params["page"] += 1
    converted_issues = []
    for issue in all_issues:
        if any([issue.get(key) is None for key in ["number", "title", "body"]]):
            logger.warning(
                f"Skipping issue {issue} as it is missing number, title, or body."
            )
            continue
        # Skip pull requests
        if "pull_request" in issue:
            continue
        converted_issues.append(
            GithubIssue(
                owner=owner,
                repo=repo,
                number=issue["number"],
                title=issue["title"],
                body=issue["body"],
            )
        )
    return converted_issues


# This function tracks the progress AND write the output to a JSONL file
async def update_progress(output: Awaitable[ResolverOutput], output_fp: TextIO, pbar: tqdm) -> None:
    resolved_output = await output
    pbar.update(1)
    pbar.set_description(f'issue {resolved_output.issue.number}')
    pbar.set_postfix_str(
        f'Test Result: {resolved_output.metrics.get("test_result", "N/A") if resolved_output.metrics else "N/A"}'
    )
    logger.info(
        f'Finished issue {resolved_output.issue.number}: {resolved_output.metrics.get("test_result", "N/A") if resolved_output.metrics else "N/A"}'
    )
    output_fp.write(resolved_output.model_dump_json() + "\n")
    output_fp.flush()


async def resolve_issues(
    owner: str,
    repo: str,
    token: str,
    username: str,
    max_iterations: int,
    limit_issues: int | None,
    num_workers: int,
    output_dir: str,
    llm_config: LLMConfig,
    runtime_container_image: str,
    prompt_template: str,  # Add this parameter
    repo_instruction: str | None,
    issue_numbers: list[int] | None,
) -> None:
    """Resolve github issues.

    Args:
        owner: Github owner of the repo.
        repo: Github repository to resolve issues in form of `owner/repo`.
        token: Github token to access the repository.
        username: Github username to access the repository.
        max_iterations: Maximum number of iterations to run
        limit_issues: Limit the number of issues to resolve.
        output_dir: Output directory to write the results.
        runtime_container_image: Container image to use.
        prompt_template: Prompt template to use.
        repo_instruction: Repository instruction to use.
        issue_numbers: List of issue numbers to resolve.
    """

    # Load dataset
    issues: list[GithubIssue] = download_issues_from_github(
        owner, repo, token
    )
    if issue_numbers is not None:
        issues = [issue for issue in issues if issue.number in issue_numbers]
        logger.info(f"Limiting resolving to issues {issue_numbers}.")
    if limit_issues is not None:
        issues = issues[:limit_issues]
        logger.info(f"Limiting resolving to first {limit_issues} issues.")

    # TEST METADATA
    model_name = llm_config.model.split("/")[-1]

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir, "infer_logs")).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f"Using output directory: {output_dir}")

    # checkout the repo
    if not os.path.exists(os.path.join(output_dir, "repo")):
        checkout_output = subprocess.check_output(
            [
            "git",
            "clone",
            f"https://{username}:{token}@github.com/{owner}/{repo}",
            f"{output_dir}/repo",
        ]
        ).decode("utf-8")
        if "fatal" in checkout_output:
            raise RuntimeError(f"Failed to clone repository: {checkout_output}")

    # get the commit id of current repo for reproducibility
    base_commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.join(output_dir, "repo")
        )
        .decode("utf-8")
        .strip()
    )
    logger.info(f"Base commit: {base_commit}")

    # OUTPUT FILE
    output_file = os.path.join(output_dir, "output.jsonl")
    logger.info(f"Writing output to {output_file}")
    finished_numbers = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                data = ResolverOutput.model_validate_json(line)
                finished_numbers.add(data.issue.number)
        logger.warning(
            f"Output file {output_file} already exists. Loaded {len(finished_numbers)} finished issues."
        )
    output_fp = open(output_file, "a")

    logger.info(
        f"Resolving issues with Agent {AGENT_CLASS}, model {model_name}, max iterations {max_iterations}."
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

    # This sets the multi-processing
    logger.info(f"Using {num_workers} workers.")

    try:
        # Replace the ProcessPoolExecutor with asyncio.gather
        tasks = []
        for issue in issues:
            task = update_progress(
                process_issue(
                    issue,
                    base_commit,
                    max_iterations,
                    llm_config,
                    output_dir,
                    runtime_container_image,
                    prompt_template,
                    repo_instruction,
                    bool(num_workers > 1),
                ),
                output_fp,
                pbar,
            )
            tasks.append(task)

        # Use asyncio.gather with a semaphore to limit concurrency
        sem = asyncio.Semaphore(num_workers)

        async def run_with_semaphore(task):
            async with sem:
                return await task

        await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Cleaning up...")
        cleanup()

    output_fp.close()
    logger.info("Finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Resolve issues from Github.")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Github repository to resolve issues in form of `owner/repo`.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Github token to access the repository.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Github username to access the repository.",
    )
    parser.add_argument(
        "--runtime-container-image",
        type=str,
        default=None,
        help="Container image to use.",
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
        "--issue-numbers",
        type=str,
        default=None,
        help="Comma separated list of issue numbers to resolve.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="workspace",
        help="Workspace directory to use.",
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
        help="LLM model to use.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key to use.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL to use.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to the prompt template file in Jinja format.",
    )
    parser.add_argument(
        "--repo-instruction-file",
        type=str,
        default=None,
        help="Path to the repository instruction file in text format.",
    )
    my_args = parser.parse_args()

    runtime_container_image = my_args.runtime_container_image
    if runtime_container_image is None:
        runtime_container_image = f"ghcr.io/all-hands-ai/runtime:oh_v{openhands.__version__}_image_nikolaik_s_python-nodejs_tag_python3.11-nodejs22"

    owner, repo = my_args.repo.split("/")
    token = (
        my_args.token if my_args.token else os.getenv("GITHUB_TOKEN")
    )
    username = (
        my_args.username
        if my_args.username
        else os.getenv("GITHUB_USERNAME")
    ) 

    if not token:
        raise ValueError("Github token is required.")

    llm_config = LLMConfig(
        model=my_args.llm_model or os.environ["LLM_MODEL"],
        api_key=my_args.llm_api_key or os.environ["LLM_API_KEY"],
        base_url=my_args.llm_base_url or os.environ.get("LLM_BASE_URL", None),
    )

    # Read the prompt template
    prompt_file = my_args.prompt_file
    if prompt_file is None:
        prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/basic-with-tests.jinja")
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    repo_instruction = None
    if my_args.repo_instruction_file:
        with open(my_args.repo_instruction_file, 'r') as f:
            repo_instruction = f.read()
    else:
        # Check for .openhands_instructions file in the workspace directory
        openhands_instructions_path = os.path.join(my_args.workspace_dir, '.openhands_instructions')
        if os.path.exists(openhands_instructions_path):
            with open(openhands_instructions_path, 'r') as f:
                repo_instruction = f.read()

    issue_numbers = None
    if my_args.issue_numbers:
        issue_numbers = [int(number) for number in my_args.issue_numbers.split(",")]

    asyncio.run(
        resolve_issues(
            owner=owner,
            repo=repo,
            token=token,
            username=username,
            runtime_container_image=runtime_container_image,
            max_iterations=my_args.max_iterations,
            limit_issues=my_args.limit_issues,
            num_workers=my_args.num_workers,
            output_dir=my_args.output_dir,
            llm_config=llm_config,
            prompt_template=prompt_template,  # Pass the prompt template
            repo_instruction=repo_instruction,
            issue_numbers=issue_numbers,
        )
    )
