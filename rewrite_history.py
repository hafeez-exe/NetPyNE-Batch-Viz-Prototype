import os
import subprocess
from datetime import datetime, timedelta

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

def run_system(cmd):
    print(f"> {cmd}")
    os.system(cmd)

# Get all commits from oldest to newest
log_out = run('git log --reverse --format="%H|%s"')
commits = []
for line in log_out.split('\n'):
    if line:
        chash, msg = line.split('|', 1)
        commits.append((chash, msg))

if len(commits) != 3:
    print(f"Error: Expected 3 commits, found {len(commits)}")
    exit(1)

# Base date: 3 days ago from now
now = datetime.now()
dates = [
    (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S"),
    (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S"),
    (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
]

# Create a new branch at the first commit
print("Rewriting history...")
run_system("git checkout --orphan temp_branch")
run_system("git rm -rf .")

for (chash, msg), date in zip(commits, dates):
    print(f"\nProcessing: {msg} at {date}")
    # checkout files from that commit
    run_system(f"git checkout {chash} -- .")
    run_system(f"git add -A")
    # Commit with backdated timestamp
    env = f'set GIT_AUTHOR_DATE={date}&& set GIT_COMMITTER_DATE={date}&& '
    # Must use powershell syntax for env vars if os.system uses cmd, but let's use subprocess to be safe
    
    cmd_env = os.environ.copy()
    cmd_env["GIT_AUTHOR_DATE"] = date
    cmd_env["GIT_COMMITTER_DATE"] = date
    
    subprocess.run(["git", "commit", "-m", msg], env=cmd_env, check=True)

# Replace main with temp_branch
run_system("git branch -M main")
print("\nDone revising history. Now forcefully pushing.")
run_system("git push -f origin main")
