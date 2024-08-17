
import argparse
import json
import os
import shutil
import subprocess
import time
from github import Github

print("Script started")

# ... (keep all the previous functions unchanged)

def main(args):
    print("Main function started")
    print(f"Current working directory: {os.getcwd()}")
    print("Environment variables:")
    for key, value in os.environ.items():
        print(f"{key}: {'*' * len(value)}")  # Print asterisks instead of the actual value for security

    issue_data = load_issue_data(args.output_dir, args.issue_number)
    if not issue_data:
        print(f"Error: Issue {args.issue_number} not found in {args.output_dir}/output.jsonl")
        return

    # ... (keep the rest of the main function unchanged)

    # Push the branch to GitHub
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set")
        return
    else:
        print(f"GITHUB_TOKEN found: {github_token[:4]}...{github_token[-4:]}")

    # ... (keep the rest of the main function unchanged)

# ... (keep the rest of the script unchanged)
