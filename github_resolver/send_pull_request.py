
import argparse
import json
import os
import shutil
import subprocess
import time
from github import Github
import re

print("Script started")

def apply_patch(repo_dir, patch):
    print(f"Applying patch to {repo_dir}")
    
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

# ... (keep the rest of the code unchanged)
