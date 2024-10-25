import re
from abc import ABC, abstractmethod
from typing import ClassVar, Any
import requests
import litellm
import jinja2
import json

from openhands.memory.history import ShortTermHistory
from openhands_resolver.github_issue import GithubIssue
from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger



class IssueHandlerInterface(ABC):
    issue_type: ClassVar[str]
    
    @abstractmethod
    def get_converted_issues(self) -> list[GithubIssue]:
        """Download issues from GitHub."""
        pass
    
    @abstractmethod
    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> str:
        """Generate instruction for the agent."""
        pass
    
    @abstractmethod
    def guess_success(self, issue: GithubIssue, history: ShortTermHistory, llm_config: LLMConfig) -> tuple[bool, list[bool] | None, str]:
        """Guess if the issue has been resolved based on the agent's output."""
        pass



    
class IssueHandler(IssueHandlerInterface):
    issue_type: ClassVar[str] = "issue"

    def __init__(self, owner: str, repo: str, token: str):
        self.download_url = "https://api.github.com/repos/{}/{}/issues"
        self.owner = owner
        self.repo = repo
        self.token = token
    
    def _download_issues_from_github(self) -> list[Any]:
        url = self.download_url.format(self.owner, self.repo)
        headers = {
            "Authorization": f"token {self.token}",
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

        return all_issues
    
    def get_converted_issues(self) -> list[GithubIssue]:
        """Download issues from Github.

        Returns:
            List of Github issues.
        """
         
        all_issues = self._download_issues_from_github()
        converted_issues = []
        for issue in all_issues:
            if any([issue.get(key) is None for key in ["number", "title", "body"]]):
                logger.warning(
                    f"Skipping issue {issue} as it is missing number, title, or body."
                )
                continue

            if "pull_request" in issue:
                continue
            
            issue_details = GithubIssue(
                                owner=self.owner,
                                repo=self.repo,
                                number=issue["number"],
                                title=issue["title"],
                                body=issue["body"],
                            )
                
            converted_issues.append(issue_details)
        return converted_issues

        
    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> str:
        """Generate instruction for the agent"""
        template = jinja2.Template(prompt_template)
        return template.render(body=issue.body, repo_instruction=repo_instruction)

    def guess_success(self, issue: GithubIssue, history: ShortTermHistory, llm_config: LLMConfig) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history and the issue description."""
       
        last_message = history.get_events_as_list()[-1].message    
        prompt = f"""Given the following issue description and the last message from an AI agent attempting to fix it, determine if the issue has been successfully resolved.

        Issue description:
        {issue.body}

        Last message from AI agent:
        {last_message}

        (1) has the issue been successfully resolved?
        (2) If the issue has been resolved, please provide an explanation of what was done in the PR that can be sent to a human reviewer on github. If the issue has not been resolved, please provide an explanation of why.

        Answer in exactly the format below, with only true or false for success, and an explanation of the result.

        --- success
        true/false

        --- explanation
        ...
        """

        response = litellm.completion(
            model=llm_config.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        
        answer = response.choices[0].message.content.strip()
        pattern = r'--- success\n*(true|false)\n*--- explanation*\n(.*)'
        match = re.search(pattern, answer)
        if match:
            return match.group(1).lower() == 'true', None, match.group(2)
        
        return False, None, f"Failed to decode answer from LLM response: {answer}"



class PRHandler(IssueHandler):
    issue_type: ClassVar[str] = "pr"

    def __init__(self, owner: str, repo: str, token: str):
        super().__init__(owner, repo, token)
        self.download_url = "https://api.github.com/repos/{}/{}/pulls"



    def __download_pr_metadata(self, pull_number: int):
    
        """
            Run a GraphQL query against the GitHub API for information on 
                1. unresolved review comments
                2. referenced issues the pull request would close

            Args:
                query: The GraphQL query as a string.
                variables: A dictionary of variables for the query.
                token: Your GitHub personal access token.

            Returns:
                The JSON response from the GitHub API.
        """
        # Using graphql as REST API doesn't indicate resolved status for review comments
        # TODO: grabbing the first 10 issues, 100 review threads, and 100 coments; add pagination to retrieve all
        query = """
                query($owner: String!, $repo: String!, $pr: Int!) {
                    repository(owner: $owner, name: $repo) {
                        pullRequest(number: $pr) {
                            closingIssuesReferences(first: 10) {
                                edges {
                                    node {
                                        body
                                    }
                                }
                            }
                            url
                            reviewThreads(first: 100) {
                                edges{
                                    node{
                                        id
                                        isResolved
                                        comments(first: 100) {
                                            totalCount
                                            nodes {
                                                body
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            """



        variables = {
            "owner": self.owner,
            "repo": self.repo,
            "pr": pull_number
        }

        url = "https://api.github.com/graphql"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        # Parse the response to get closing issue references and unresolved review comments
        pr_data = response_json.get("data", {}).get("repository", {}).get("pullRequest", {})

        # Get closing issues
        closing_issues = pr_data.get("closingIssuesReferences", {}).get("edges", [])
        closing_issues_bodies = [issue["node"]["body"] for issue in closing_issues]

        # Get unresolved review comments
        unresolved_comments = []
        thread_ids = []  # Store comment thread IDs; agent replies to the thread
        review_threads = pr_data.get("reviewThreads", {}).get("edges", [])
        for thread in review_threads:
            node = thread.get("node", {})
            if not node.get("isResolved", True):  # Check if the review thread is unresolved
                id = node.get("id")
                thread_ids.append(id)
                comments = node.get("comments", {}).get("nodes", [])
                message = ""
                for i, comment in enumerate(comments):
                    if i == len(comments) - 1:  # Check if it's the last comment in the thread
                        if len(comments) > 1:
                            message += "---\n"  # Add "---" before the last message if there's more than one comment
                        message += "latest feedback:\n" + comment["body"] + "\n"
                    else:
                        message += comment["body"] + "\n"  # Add each comment in a new line
                unresolved_comments.append(message)

        return closing_issues_bodies, unresolved_comments, thread_ids


    # Override processing of downloaded issues
    def get_converted_issues(self) -> list[GithubIssue]:
        all_issues = self._download_issues_from_github()
        converted_issues = []
        for issue in all_issues:
            if any([issue.get(key) is None for key in ["number", "title", "body"]]):
                logger.warning(
                    f"Skipping issue {issue} as it is missing number, title, or body."
                )
                continue            

            closing_issues, unresolved_comments, thread_ids = self.__download_pr_metadata(issue["number"])
            head_branch = issue["head"]["ref"]
            issue_details = GithubIssue(
                                owner=self.owner,
                                repo=self.repo,
                                number=issue["number"],
                                title=issue["title"],
                                body=issue["body"],
                                closing_issues=closing_issues,
                                review_comments=unresolved_comments,
                                thread_ids=thread_ids,
                                head_branch=head_branch
                            )
            
            converted_issues.append(issue_details)

        return converted_issues


    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> str:
        """Generate instruction for the agent"""
        template = jinja2.Template(prompt_template)

        if issue.closing_issues is None or issue.review_comments is None:
            raise ValueError("issue.closing_issues or issue.review_comments is None")
        issues_context = json.dumps(issue.closing_issues, indent=4)
        comment_chain = json.dumps(issue.review_comments, indent=4)
        instruction = template.render(issues=issues_context, body=comment_chain, repo_instruction=repo_instruction)
        return instruction
    

    def guess_success(self, issue: GithubIssue, history: ShortTermHistory, llm_config: LLMConfig) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history and the issue description."""
        
        last_message = history.get_events_as_list()[-1].message
        issues_context = json.dumps(issue.closing_issues, indent=4)
        success_list = []
        explanation_list = []

        if issue.review_comments:
            for comment in issue.review_comments:
                formatted_comment = json.dumps(comment, indent=4)
                prompt = f"""You are given one or more issue descriptions, a piece of feedback to resolve the issues, and the last message from an AI agent attempting to incorporate the feedback. Determine if the feedback has been successfully resolved.
                
                Issue descriptions:
                {issues_context}

                Feedback:
                {formatted_comment}

                Last message from AI agent:
                {last_message}

                (1) has the feedback been successfully incorporated?
                (2) If the feebdack has been incorporated, please provide an explanation of what was done that can be sent to a human reviewer on github. If the feedback has not been resolved, please provide an explanation of why.

                Answer in exactly the format below, with only true or false for success, and an explanation of the result.

                --- success
                true/false

                --- explanation
                ...
                """

                response = litellm.completion(
                    model=llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=llm_config.api_key,
                    base_url=llm_config.base_url,
                )
            
                answer = response.choices[0].message.content.strip()
                pattern = r'--- success\n*(true|false)\n*--- explanation*\n(.*)'
                match = re.search(pattern, answer)
                if match:
                    success_list.append(match.group(1).lower() == 'true')
                    explanation_list.append(match.group(2))
                else:
                    success_list.append(False)
                    f"Failed to decode answer from LLM response: {answer}"
        else:
            raise ValueError("Expected review comments to be initialized.")
        
        success = all(success_list)
        return success, success_list, json.dumps(explanation_list)