#!/usr/bin/env bash
set -euo pipefail

# Ralph HITL - Single interactive iteration (human-in-the-loop)
# Usage: ./plans/ralph-once.sh [--llm claude|codex]
#
# Runs one iteration of the Ralph loop interactively so you can steer and review.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plans/ralph_common.sh
source "$SCRIPT_DIR/ralph_common.sh"

usage() {
  cat <<'EOF'
Usage: ./plans/ralph-once.sh [--llm claude|codex]

Examples:
  ./plans/ralph-once.sh
  ./plans/ralph-once.sh --llm codex
  RALPH_LLM=codex ./plans/ralph-once.sh
EOF
  echo ""
  ralph_print_provider_help
}

PROVIDER="${RALPH_LLM:-claude}"

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
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PROVIDER="$(ralph_resolve_provider "$PROVIDER")"
PROJECT_DIR="$(ralph_project_dir)"
PROMPT="$(ralph_load_prompt)"

cd "$PROJECT_DIR"

echo "Starting Ralph HITL - single interactive iteration"
echo "LLM provider: $PROVIDER"
echo "=========================================="

ralph_run_interactive "$PROVIDER" "$PROMPT"
