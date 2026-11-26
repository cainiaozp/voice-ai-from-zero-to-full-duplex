#!/usr/bin/env bash
# Create and push the course repository with GitHub CLI.
# Authentication is performed in the user's own environment via gh auth login.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="${1:-voice-ai-from-zero-to-full-duplex}"
VISIBILITY="${GH_REPO_VISIBILITY:-public}"
DRY_RUN="${2:-}"

if [[ "$REPO_NAME" == '--dry-run' ]]; then
    REPO_NAME='voice-ai-from-zero-to-full-duplex'
    DRY_RUN='--dry-run'
fi
if [[ "$VISIBILITY" != 'public' && "$VISIBILITY" != 'private' ]]; then
    echo 'GH_REPO_VISIBILITY must be public or private' >&2
    exit 2
fi
cd "$ROOT"

if [[ "$DRY_RUN" == '--dry-run' ]]; then
    echo "Directory: $ROOT"
    echo "Repository: $REPO_NAME ($VISIBILITY)"
    echo 'Would: validate files; initialize git; commit; gh repo create --source=. --push'
    exit 0
fi

command -v git >/dev/null || { echo 'Install Git first.' >&2; exit 1; }
command -v gh >/dev/null || { echo 'GitHub CLI not installed: https://cli.github.com/' >&2; exit 1; }
if ! gh auth status >/dev/null 2>&1; then
    echo 'Authorize GitHub using: gh auth login --hostname github.com --git-protocol https --web' >&2
    echo 'Then rerun this script. Do not use your GitHub account password.' >&2
    exit 1
fi

if command -v python >/dev/null; then
    python scripts/verify_repo.py
elif command -v python3 >/dev/null; then
    python3 scripts/verify_repo.py
else
    echo 'Python is required to validate the course files.' >&2
    exit 1
fi

if [[ ! -d .git ]]; then
    git init -b main
fi

# Set a local-only Git identity if none is configured.
# This uses GitHub's no-reply address instead of exposing a private email address.
if [[ -z "$(git config --get user.name || true)" ]]; then
    git config --local user.name "$(gh api user --jq '.login')"
fi
if [[ -z "$(git config --get user.email || true)" ]]; then
    GH_ID="$(gh api user --jq '.id')"
    GH_LOGIN="$(gh api user --jq '.login')"
    git config --local user.email "${GH_ID}+${GH_LOGIN}@users.noreply.github.com"
fi

git add .
echo 'Files to be included (summary):'
git diff --cached --stat | tail -n 8 || true

if ! git diff --cached --quiet; then
    git commit -m 'docs: publish 10-part Voice AI course and reproducible labs'
fi

if git remote get-url origin >/dev/null 2>&1; then
    echo 'Existing origin remote found; pushing current branch.'
    gh auth setup-git
    git push -u origin HEAD
else
    gh repo create "$REPO_NAME" "--${VISIBILITY}" --source=. --remote=origin --push
fi

echo 'Published repository:'
gh repo view --json url --jq '.url'
