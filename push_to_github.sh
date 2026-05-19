#!/bin/bash
# Push repo to GitHub server - run with: GITHUB_TOKEN=xxx ./push_to_github.sh
# Or set GH_TOKEN/GITHUB_TOKEN env var first

# Usage: GITHUB_TOKEN=ghp_xxx ./push_to_github.sh
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
    echo "ERROR: Set GH_TOKEN or GITHUB_TOKEN environment variable"
    exit 1
fi

cd "$(dirname "$0")"
git remote set-url origin "https://oauth2:${TOKEN}@github.com/huisuon254-netizen/k234.git"
git push -u origin main 2>&1
git remote set-url origin "https://github.com/huisuon254-netizen/k234.git"
echo "Push complete!"
