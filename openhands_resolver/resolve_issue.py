# flake8: noqa: E501

import asyncio
import dataclasses
import shutil
import os
import pathlib
import subprocess
import json
import argparse
from typing import Any, TextIO

from termcolor import colored
from tqdm import tqdm

from openhands_resolver.github_issue import GithubIssue
from openhands_resolver.issue_definitions import (
    IssueHandler,
    IssueHandlerInterface,
    PRHandler
)
from openhands_resolver.resolver_output import ResolverOutput
import openhands
from openhands.core.main import create_runtime, run_controller
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    LLMConfig,
)
from openhands.runtime.runtime import Runtime
from openhands_resolver.utils import (
    codeact_user_response,
    reset_logger_for_multiprocessing,
)

# Don't make this confgurable for now, unless we have other competitive agents
AGENT_CLASS = "CodeActAgent"

def initialize_runtime(runtime: Runtime):
    """Initialize the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to change directory to /workspace.\n{obs}")

    action = CmdRunAction(command='git config --global core.pager ""')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to set git config.\n{obs}")

async def complete_runtime(runtime: Runtime, base_commit: str) -> dict[str, Any]:
    """Complete the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: Observation

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to change directory to /workspace. Observation: {obs}")

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

async def process_issue(
    issue: GithubIssue,
    base_commit: str,
    max_iterations: int,
    llm_config: LLMConfig,
    output_dir: str,
    runtime_container_image: str,
    prompt_template: str,
    issue_handler: IssueHandlerInterface,
    repo_instruction: str | None = None,
    reset_logger: bool = True,
) -> ResolverOutput:
    """Process a single issue."""
    # Setup the logger properly
    if reset_logger:
        log_dir = os.path.join(output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
    else:
        logger.info(f'Starting fixing issue {issue.number}.')

    workspace_base = os.path.join(output_dir, "workspace", f"{issue_handler.issue_type}_{issue.number}")

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
            timeout=300,
        ),
        workspace_base=workspace_base,
        workspace_mount_path=workspace_base,
    )
    config.set_llm_config(llm_config)

    runtime = create_runtime(config, sid=f"{issue.number}")
    initialize_runtime(runtime)

    instruction = issue_handler.get_instruction(issue, prompt_template, repo_instruction)
    action = MessageAction(content=instruction)
    state: State | None = await run_controller(
        config=config,
        initial_user_action=action,
        runtime=runtime,
        fake_user_response_fn=codeact_user_response,
    )
    if state is None:
        raise RuntimeError("Failed to run the agent.")

    # Get git patch
    return_val = await complete_runtime(runtime, base_commit)
    git_patch = return_val['git_patch']
    logger.info(f'Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------')

    # Serialize histories
    histories = [dataclasses.asdict(event) for event in state.history.get_events()]
    metrics = state.metrics.get() if state.metrics else None

    # determine success based on the history and the issue description
    success, comment_success, success_explanation = issue_handler.guess_success(issue, state.history, llm_config)

    if issue_handler.issue_type == "pr" and comment_success:
        success_log = "I have updated the PR and resolved some of the issues that were cited in the pull request review. Specifically, I identified the following revision requests, and all the ones that I think I successfully resolved are checked off. All the unchecked ones I was not able to resolve, so manual intervention may be required:\n"
        for success_indicator, explanation in zip(comment_success, json.loads(success_explanation)):
                status = colored("[X]", "red") if success_indicator else colored("[ ]", "red")
                bullet_point = colored("-", "yellow")
                success_log += f"\n{bullet_point} {status}: {explanation}"
        logger.info(success_log)

    # Save the output
    output = ResolverOutput(
        issue=issue,
        issue_type=issue_handler.issue_type,
        instruction=instruction,
        base_commit=base_commit,
        git_patch=git_patch,
        history=histories,
        metrics=metrics,
        success=success,
        comment_success=comment_success,
        success_explanation=success_explanation,
        error=state.last_error if state and state.last_error else None,
    )
    return output

def issue_handler_factory(issue_type: str, owner: str, repo: str, token: str) -> IssueHandlerInterface:
    """Create an issue handler based on the issue type."""
    if issue_type == "issue":
        return IssueHandler(owner, repo, token)
    elif issue_type == "pr":
        return PRHandler(owner, repo, token)
    else:
        raise ValueError(f"Invalid issue type: {issue_type}")

async def resolve_issue(
    owner: str,
    repo: str,
    token: str,
    username: str,
    issue_number: int,
    max_iterations: int,
    output_dir: str,
    llm_config: LLMConfig,
    runtime_container_image: str,
    prompt_template: str,
    issue_type: str,
    repo_instruction: str | None,
) -> None:
    """Resolve a single github issue.

    Args:
        owner: Github owner of the repo.
        repo: Github repository name.
        token: Github token to access the repository.
        username: Github username to access the repository.
        issue_number: Issue number to resolve.
        max_iterations: Maximum number of iterations to run.
        output_dir: Output directory to write the results.
        runtime_container_image: Container image to use.
        prompt_template: Prompt template to use.
        issue_type: Type of issue to resolve (issue or pr).
        repo_instruction: Repository instruction to use.
    """
    issue_handler = issue_handler_factory(issue_type, owner, repo, token)

    # Load dataset
    issues: list[GithubIssue] = issue_handler.get_converted_issues()
    issues = [issue for issue in issues if issue.number == issue_number]
    if not issues:
        raise ValueError(f"Issue {issue_number} not found.")
    issue = issues[0]

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir, "infer_logs")).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # checkout the repo
    repo_dir = os.path.join(output_dir, "repo")
    if not os.path.exists(repo_dir):
        checkout_output = subprocess.check_output([
            "git",
            "clone",
            f"https://{username}:{token}@github.com/{owner}/{repo}",
            f"{output_dir}/repo",
        ]).decode("utf-8")
        if "fatal" in checkout_output:
            raise RuntimeError(f"Failed to clone repository: {checkout_output}")

    # get the commit id of current repo for reproducibility
    base_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        .decode("utf-8")
        .strip()
    )
    logger.info(f"Base commit: {base_commit}")

    if repo_instruction is None:
        # Check for .openhands_instructions file in the workspace directory
        openhands_instructions_path = os.path.join(repo_dir, '.openhands_instructions')
        if os.path.exists(openhands_instructions_path):
            with open(openhands_instructions_path, 'r') as f:
                repo_instruction = f.read()

    # OUTPUT FILE
    output_file = os.path.join(output_dir, "output.jsonl")
    logger.info(f"Writing output to {output_file}")

    # Process the issue
    output = await process_issue(
        issue,
        base_commit,
        max_iterations,
        llm_config,
        output_dir,
        runtime_container_image,
        prompt_template,
        issue_handler,
        repo_instruction,
    )

    # Write output
    with open(output_file, "a") as output_fp:
        output_fp.write(output.model_dump_json() + "\n")

    logger.info("Finished.")

def main():
    parser = argparse.ArgumentParser(description="Resolve a single issue from Github.")
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
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of iterations to run.",
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        required=True,
        help="Issue number to resolve.",
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
    parser.add_argument(
        "--issue-type",
        type=str,
        default="issue",
        choices=["issue", "pr"],
        help="Type of issue to resolve, either open issue or pr comments.",
    )

    my_args = parser.parse_args()

    runtime_container_image = my_args.runtime_container_image
    if runtime_container_image is None:
        runtime_container_image = f"ghcr.io/all-hands-ai/runtime:{openhands.__version__}-nikolaik"

    owner, repo = my_args.repo.split("/")
    token = my_args.token if my_args.token else os.getenv("GITHUB_TOKEN")
    username = my_args.username if my_args.username else os.getenv("GITHUB_USERNAME")

    if not token:
        raise ValueError("Github token is required.")

    llm_config = LLMConfig(
        model=my_args.llm_model or os.environ["LLM_MODEL"],
        api_key=my_args.llm_api_key or os.environ["LLM_API_KEY"],
        base_url=my_args.llm_base_url or os.environ.get("LLM_BASE_URL", None),
    )

    repo_instruction = None
    if my_args.repo_instruction_file:
        with open(my_args.repo_instruction_file, 'r') as f:
            repo_instruction = f.read()

    issue_type = my_args.issue_type

    # Read the prompt template
    prompt_file = my_args.prompt_file
    if prompt_file is None:
        if issue_type == "issue":
            prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/basic-with-tests.jinja")
        else:
            prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/basic-followup.jinja") 
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    asyncio.run(
        resolve_issue(
            owner=owner,
            repo=repo,
            token=token,
            username=username,
            issue_number=my_args.issue_number,
            max_iterations=my_args.max_iterations,
            output_dir=my_args.output_dir,
            llm_config=llm_config,
            runtime_container_image=runtime_container_image,
            prompt_template=prompt_template,
            issue_type=issue_type,
            repo_instruction=repo_instruction,
        )
    )


if __name__ == "__main__":
    main()

