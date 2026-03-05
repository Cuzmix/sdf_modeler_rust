#!/usr/bin/env bash
set -euo pipefail

# Ralph HITL — Single interactive iteration (human-in-the-loop)
# Usage: ./plans/ralph-once.sh
#
# Runs one iteration of the Ralph loop interactively so you can steer and review.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "Starting Ralph HITL — single interactive iteration"
echo "=========================================="

claude \
  "$(cat <<'PROMPT'
You are working through a product requirements document (PRD) for this project.

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
output exactly: <promise>COMPLETE</promise>
PROMPT
  )"
