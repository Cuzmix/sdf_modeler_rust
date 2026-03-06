#!/usr/bin/env bash
set -euo pipefail

# Ralph AFK — Autonomous coding agent loop
# Usage: ./plans/ralph.sh <max_iterations>
#
# Runs Claude Code in a headless loop, working through the PRD one task at a time.
# Each iteration: pick a task, implement it, verify, commit, repeat.

if [ -z "${1:-}" ]; then
  echo "Usage: ./plans/ralph.sh <max_iterations>"
  echo "Example: ./plans/ralph.sh 10"
  exit 1
fi

MAX_ITERATIONS=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

PROMPT='You are working through a product requirements document (PRD) for this project.

@plans/prd.json contains the task list. Each item has a "passes" flag.
@plans/progress.txt contains notes from previous iterations.

Follow these steps:

1. Read plans/prd.json and plans/progress.txt to understand current state.
2. Find the highest-priority feature that has "passes": false. Prioritize by
   importance and dependencies — not necessarily the first item in the list.
3. Implement ONLY that single feature. Keep the change small and focused.
4. Run the verification loops:
   - cargo check
   - cargo clippy -- -D warnings
   - cargo test
   Fix any failures before proceeding.
5. Update plans/prd.json — set "passes": true for completed items.
6. APPEND your progress to plans/progress.txt (do NOT overwrite).
   Include: what you did, files changed, any notes for the next iteration.
7. Make a git commit with a descriptive message. Include all changed files
   (code + prd.json + progress.txt).
8. Only work on a SINGLE feature per iteration.

If while implementing you notice the PRD is complete (all items pass),
output exactly: <promise>COMPLETE</promise>'

echo "Starting Ralph AFK — max $MAX_ITERATIONS iterations"
echo "Project: $PROJECT_DIR"
echo "=========================================="

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  echo ""
  echo "========== Ralph iteration $i / $MAX_ITERATIONS =========="
  echo ""

  result=$(echo "$PROMPT" | claude -p --allowedTools "Edit,Write,Read,Glob,Grep,Bash,TodoWrite")

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo ""
    echo "========== Ralph COMPLETE after $i iterations =========="
    exit 0
  fi
done

echo ""
echo "========== Ralph reached max iterations ($MAX_ITERATIONS) =========="
