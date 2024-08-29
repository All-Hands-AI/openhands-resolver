
import os
from github import Github
from github_resolver.resolve_issues import download_issue_info, process_issue
from typing import List, Dict

def update_pull_request(repo_name: str, pr_number: int) -> None:
    """
    Update a pull request based on review comments.
    
    :param repo_name: The name of the repository
    :param pr_number: The number of the pull request
    """
    # Initialize Github client
    github_token = os.getenv('GITHUB_TOKEN')
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    
    # Get the pull request
    pr = repo.get_pull(pr_number)
    
    # Download associated issue information
    issues = download_associated_issues(pr)
    
    # Download review comments
    review_comments = download_review_comments(pr)
    
    # Process unsolved comments
    unsolved_comments = process_unsolved_comments(review_comments)
    
    # Make modifications to the codebase
    make_modifications(repo, pr, unsolved_comments)
    
    # Check if comments have been resolved
    resolved_comments = check_resolved_comments(pr, unsolved_comments)
    
    # Print summary
    print(f"Total comments: {len(review_comments)}")
    print(f"Resolved comments: {len(resolved_comments)}")
    print(f"Unresolved comments: {len(unsolved_comments) - len(resolved_comments)}")

def download_associated_issues(pr) -> List[Dict]:
    """Download information from issues associated with the PR"""
    issues = []
    body = pr.body
    issue_numbers = [int(s) for s in body.split() if s.startswith('#') and s[1:].isdigit()]
    
    for issue_number in issue_numbers:
        try:
            issue = pr.base.repo.get_issue(issue_number)
            issue_info = download_issue_info(issue)
            issues.append(issue_info)
        except Exception as e:
            print(f"Error downloading issue #{issue_number}: {str(e)}")
    
    return issues

def download_review_comments(pr) -> List[Dict]:
    """Download all review comments from the PR"""
    review_comments = []
    for comment in pr.get_review_comments():
        review_comments.append({
            'id': comment.id,
            'body': comment.body,
            'user': comment.user.login,
            'created_at': comment.created_at,
            'path': comment.path,
            'position': comment.position
        })
    return review_comments

def process_unsolved_comments(comments: List[Dict]) -> List[Dict]:
    """Process and return unsolved comments"""
    unsolved_comments = []
    for comment in comments:
        if not comment.get('resolved', False):
            unsolved_comments.append(comment)
    return unsolved_comments

def make_modifications(repo, pr, unsolved_comments: List[Dict]) -> None:
    """Make appropriate modifications to the codebase"""
    for comment in unsolved_comments:
        file_path = comment['path']
        position = comment['position']
        suggestion = extract_suggestion(comment['body'])
        
        if suggestion:
            try:
                file_content = repo.get_contents(file_path, ref=pr.head.ref).decoded_content.decode('utf-8')
                lines = file_content.splitlines()
                lines[position - 1] = suggestion
                updated_content = '\n'.join(lines)
                
                commit_message = f"Address review comment: {comment['id']}"
                repo.update_file(file_path, commit_message, updated_content, repo.get_contents(file_path, ref=pr.head.ref).sha, branch=pr.head.ref)
                print(f"Modified file {file_path} at line {position}")
            except Exception as e:
                print(f"Error modifying file {file_path}: {str(e)}")

def extract_suggestion(comment_body: str) -> str:
    """Extract code suggestion from comment body"""
    # This is a simple implementation. You might want to improve it based on your specific needs.
    lines = comment_body.split('\n')
    suggestion_lines = []
    in_suggestion = False
    for line in lines:
        if line.strip().startswith('```'):
            in_suggestion = not in_suggestion
        elif in_suggestion:
            suggestion_lines.append(line)
    return '\n'.join(suggestion_lines) if suggestion_lines else None

def check_resolved_comments(pr, unsolved_comments: List[Dict]) -> None:
    """Check if comments have been successfully resolved"""
    resolved_comments = []
    for comment in unsolved_comments:
        file_path = comment['path']
        position = comment['position']
        try:
            file_content = pr.get_files()[0].patch
            lines = file_content.splitlines()
            if position <= len(lines) and extract_suggestion(comment['body']) in lines[position - 1]:
                resolved_comments.append(comment)
                print(f"Comment {comment['id']} has been resolved")
            else:
                print(f"Comment {comment['id']} is still unresolved")
        except Exception as e:
            print(f"Error checking comment {comment['id']}: {str(e)}")
    
    return resolved_comments

if __name__ == "__main__":
    repo_name = "example/repo"
    pr_number = 1
    update_pull_request(repo_name, pr_number)
