#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

if [[ ! -f "${repo_root}/.githooks/pre-commit" || ! -f "${repo_root}/.githooks/commit-msg" ]]; then
  echo "Expected hooks in ${repo_root}/.githooks were not found."
  exit 1
fi

chmod +x "${repo_root}/.githooks/pre-commit" "${repo_root}/.githooks/commit-msg"

git config core.hooksPath ".githooks"
git config commit.template ".gitmessage-logical-unit.txt"

echo "Logical commit workflow enabled for: ${repo_root}"
echo "  core.hooksPath   = $(git config --get core.hooksPath)"
echo "  commit.template  = $(git config --get commit.template)"
echo
echo "Hook limits:"
echo "  LOGICAL_COMMIT_MAX_FILES=${LOGICAL_COMMIT_MAX_FILES:-20}"
echo "  LOGICAL_COMMIT_MAX_LINES=${LOGICAL_COMMIT_MAX_LINES:-600}"
echo
echo "Temporary bypass (for exceptional large commits):"
echo "  LOGICAL_COMMIT_BYPASS=1 git commit ..."
