import re
from abc import ABC, abstractmethod
from typing import ClassVar, Any
import requests
import litellm
import jinja2
import json

from openhands.events.event import Event
from openhands_resolver.github_issue import GithubIssue, ReviewThread
from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger



class IssueHandlerInterface(ABC):
    issue_type: ClassVar[str]
    
    @abstractmethod
    def get_converted_issues(self) -> list[GithubIssue]:
        """Download issues from GitHub."""
        pass
    
    @abstractmethod
    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> tuple[str, list[str]]:
        """Generate instruction and image urls for the agent."""
        pass
    
    @abstractmethod
    def guess_success(self, issue: GithubIssue, history: list[Event], llm_config: LLMConfig) -> tuple[bool, list[bool] | None, str]:
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
    def _extract_image_urls(self, issue_body: str) -> list[str]:
        # Regular expression to match Markdown image syntax ![alt text](image_url)
        image_pattern = r'!\[.*?\]\((https?://[^\s)]+)\)'
        return re.findall(image_pattern, issue_body)

    def _get_issue_comments(self, issue_number: int) -> list[str] | None:
        """Download comments for a specific issue from Github."""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        params = {"per_page": 100, "page": 1}
        all_comments = []

        while True:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            comments = response.json()

            if not comments:
                break

            all_comments.extend([comment["body"] for comment in comments])
            params["page"] += 1

        return all_comments if all_comments else None
    
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
            
            # Get issue thread comments
            thread_comments = self._get_issue_comments(issue["number"])
            # Convert empty lists to None for optional fields
            issue_details = GithubIssue(
                                owner=self.owner,
                                repo=self.repo,
                                number=issue["number"],
                                title=issue["title"],
                                body=issue["body"],
                                thread_comments=thread_comments,
                                review_comments=None,  # Initialize review comments as None for regular issues
                            )
                
            converted_issues.append(issue_details)
        return converted_issues

    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> tuple[str, list[str]]:
        """Generate instruction for the agent"""
        # Format thread comments if they exist
        thread_context = ""
        if issue.thread_comments:
            thread_context = "\n\nIssue Thread Comments:\n" + "\n---\n".join(issue.thread_comments)
        
        images = []
        images.extend(self._extract_image_urls(issue.body))
        images.extend(self._extract_image_urls(thread_context))

        template = jinja2.Template(prompt_template)
        return template.render(body=issue.body + thread_context, repo_instruction=repo_instruction), images




    def guess_success(self, issue: GithubIssue, history: list[Event], llm_config: LLMConfig) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history and the issue description."""
       
        last_message = history[-1].message
        # Include thread comments in the prompt if they exist
        issue_context = issue.body
        if issue.thread_comments:
            issue_context += "\n\nIssue Thread Comments:\n" + "\n---\n".join(issue.thread_comments)
            
        prompt = f"""Given the following issue description and the last message from an AI agent attempting to fix it, determine if the issue has been successfully resolved.

        Issue description:
        {issue_context}

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



    def __download_pr_metadata(self, pull_number: int) -> tuple[list[str], list[str], list[ReviewThread], list[str]]:
    
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
                            reviews(first: 100) {
                                nodes {
                                    body
                                    state
                                }
                            }
                            reviewThreads(first: 100) {
                                edges{
                                    node{
                                        id
                                        isResolved
                                        comments(first: 100) {
                                            totalCount
                                            nodes {
                                                body
                                                path
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

        # Get review comments
        reviews = pr_data.get("reviews", {}).get("nodes", [])
        review_bodies = [review["body"] for review in reviews]

        # Get unresolved review threads
        review_threads = []
        thread_ids = []  # Store thread IDs; agent replies to the thread
        raw_review_threads = pr_data.get("reviewThreads", {}).get("edges", [])
        for thread in raw_review_threads:
            node = thread.get("node", {})
            if not node.get("isResolved", True):  # Check if the review thread is unresolved
                id = node.get("id")
                thread_ids.append(id)
                my_review_threads = node.get("comments", {}).get("nodes", [])
                message = ""
                files = []
                for i, review_thread in enumerate(my_review_threads):
                    if i == len(my_review_threads) - 1:  # Check if it's the last thread in the thread
                        if len(my_review_threads) > 1:
                            message += "---\n"  # Add "---" before the last message if there's more than one thread
                        message += "latest feedback:\n" + review_thread["body"] + "\n"
                    else:
                        message += review_thread["body"] + "\n"  # Add each thread in a new line
                    
                    file = review_thread.get("path")
                    if file:
                        files.append(file)

                unresolved_thread = ReviewThread(
                    comment=message,
                    files=files
                )
                review_threads.append(unresolved_thread)

        return closing_issues_bodies, review_bodies, review_threads, thread_ids


    # Override processing of downloaded issues
    def get_converted_issues(self) -> list[GithubIssue]:
        all_issues = self._download_issues_from_github()
        converted_issues = []
        for issue in all_issues:
            # For PRs, body can be None
            if any([issue.get(key) is None for key in ["number", "title"]]):
                logger.warning(
                    f"Skipping #{issue} as it is missing number or title."
                )
                continue            

            # Handle None body for PRs
            body = issue.get("body") if issue.get("body") is not None else ""
            closing_issues, review_comments, review_threads, thread_ids = self.__download_pr_metadata(issue["number"])
            head_branch = issue["head"]["ref"]
            issue_details = GithubIssue(
                                owner=self.owner,
                                repo=self.repo,
                                number=issue["number"],
                                title=issue["title"],
                                body=body,
                                closing_issues=closing_issues,
                                review_comments=review_comments,
                                review_threads=review_threads,
                                thread_ids=thread_ids,
                                head_branch=head_branch
                            )
            
            converted_issues.append(issue_details)

        return converted_issues


    def get_instruction(self, issue: GithubIssue, prompt_template: str, repo_instruction: str | None = None) -> tuple[str, list[str]]:
        """Generate instruction for the agent"""
        template = jinja2.Template(prompt_template)
        images = []

        issues_str = None
        if issue.closing_issues:
            issues_str = json.dumps(issue.closing_issues, indent=4) 
            images.extend(self._extract_image_urls(issues_str))

        # Handle PRs with review comments
        review_comments_str = None
        if issue.review_comments:
            review_comments_str = json.dumps(issue.review_comments, indent=4)
            images.extend(self._extract_image_urls(review_comments_str))

        # Handle PRs with file-specific review comments
        review_thread_str = None
        review_thread_file_str = None
        if issue.review_threads:
            review_threads = [review_thread.comment for review_thread in issue.review_threads]
            review_thread_files = []
            for review_thread in issue.review_threads:
                review_thread_files.extend(review_thread.files)
            review_thread_str = json.dumps(review_threads, indent=4)
            review_thread_file_str = json.dumps(review_thread_files, indent=4)
            images.extend(self._extract_image_urls(review_thread_str))
        

        instruction = template.render(issues=issues_str, review_comments=review_comments_str, review_threads=review_thread_str, files=review_thread_file_str, repo_instruction=repo_instruction)
        return instruction, images
    

    def guess_success(self, issue: GithubIssue, history: list[Event], llm_config: LLMConfig) -> tuple[bool, None | list[bool], str]:
        """Guess if the issue is fixed based on the history and the issue description."""
        
        last_message = history[-1].message
        issues_context = json.dumps(issue.closing_issues, indent=4)
        success_list = []
        explanation_list = []

        # Handle PRs with file-specific review comments
        if issue.review_threads is not None:
            for review_thread in issue.review_threads:
                files_context = json.dumps(review_thread.files, indent=4)

                prompt = f"""You are given one or more issue descriptions, a piece of feedback to resolve the issues, and the last message from an AI agent attempting to incorporate the feedback. If the feedback is addressed to a specific code file, then the file locations will be provided as well. Determine if the feedback has been successfully resolved.
                
                Issue descriptions:
                {issues_context}

                Feedback:
                {review_thread.comment}

                Files locations:
                {files_context}

                Last message from AI agent:
                {last_message}

                (1) has the feedback been successfully incorporated?
                (2) If the feedback has been incorporated, please provide an explanation of what was done that can be sent to a human reviewer on github. If the feedback has not been resolved, please provide an explanation of why.

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
        # Handle PRs with only thread comments (no file-specific review comments)
        elif issue.thread_comments:
            thread_context = "\n---\n".join(issue.thread_comments)
            prompt = f"""You are given one or more issue descriptions, the PR thread comments, and the last message from an AI agent attempting to address the feedback. Determine if the feedback has been successfully resolved.
            
            Issue descriptions:
            {issues_context}

            PR Thread Comments:
            {thread_context}

            Last message from AI agent:
            {last_message}

            (1) has the feedback been successfully incorporated?
            (2) If the feedback has been incorporated, please provide an explanation of what was done that can be sent to a human reviewer on github. If the feedback has not been resolved, please provide an explanation of why.

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
                explanation_list.append(f"Failed to decode answer from LLM response: {answer}")
        else:
            # No review comments or thread comments found
            raise ValueError("Expected review comments or thread comments to be initialized.")
            
        # Return overall success (all must be true) and explanations
        if not success_list:
            return False, None, "No feedback was processed"
        return all(success_list), success_list, "\n".join(explanation_list)












