#!/usr/bin/env bash
set -euo pipefail

# Ralph AFK - Autonomous coding agent loop
# Usage: ./plans/ralph.sh <max_iterations> [--llm claude|codex]
#
# Runs the selected coding agent in a headless loop, working through the PRD one
# task at a time. Each iteration: pick a task, implement it, verify, commit,
# repeat.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plans/ralph_common.sh
source "$SCRIPT_DIR/ralph_common.sh"

usage() {
  cat <<'EOF'
Usage: ./plans/ralph.sh <max_iterations> [--llm claude|codex]

Examples:
  ./plans/ralph.sh 10
  ./plans/ralph.sh 10 --llm codex
  RALPH_LLM=codex ./plans/ralph.sh 5
EOF
  echo ""
  ralph_print_provider_help
}

PROVIDER="${RALPH_LLM:-claude}"
POSITIONAL_ARGS=()

while (($# > 0)); do
  case "$1" in
    --llm)
      if [ -z "${2:-}" ]; then
        echo "Missing value for --llm" >&2
        usage
        exit 1
      fi
      PROVIDER="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${#POSITIONAL_ARGS[@]}" -ne 1 ]; then
  usage
  exit 1
fi

MAX_ITERATIONS="${POSITIONAL_ARGS[0]}"

if ! [[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_iterations must be a positive integer." >&2
  exit 1
fi

PROVIDER="$(ralph_resolve_provider "$PROVIDER")"
PROJECT_DIR="$(ralph_project_dir)"
PROMPT="$(ralph_load_prompt)"

cd "$PROJECT_DIR"

echo "Starting Ralph AFK - max $MAX_ITERATIONS iterations"
echo "Project: $PROJECT_DIR"
echo "LLM provider: $PROVIDER"
echo "=========================================="

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  echo ""
  echo "========== Ralph iteration $i / $MAX_ITERATIONS =========="
  echo ""

  result="$(ralph_run_headless "$PROVIDER" "$PROMPT")"

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo ""
    echo "========== Ralph COMPLETE after $i iterations =========="
    exit 0
  fi
done

echo ""
echo "========== Ralph reached max iterations ($MAX_ITERATIONS) =========="
