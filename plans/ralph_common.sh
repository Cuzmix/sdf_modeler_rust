#!/usr/bin/env bash

RALPH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RALPH_PROJECT_DIR="$(cd "$RALPH_SCRIPT_DIR/.." && pwd)"
RALPH_PROMPT_FILE="$RALPH_SCRIPT_DIR/ralph_prompt.txt"
RALPH_CODEX_WRAPPER="$RALPH_SCRIPT_DIR/ralph_codex.ps1"
RALPH_DEFAULT_PROVIDER="claude"
RALPH_CLAUDE_ALLOWED_TOOLS="Edit,Write,Read,Glob,Grep,Bash,TodoWrite"

ralph_project_dir() {
  printf '%s\n' "$RALPH_PROJECT_DIR"
}

ralph_load_prompt() {
  cat "$RALPH_PROMPT_FILE"
}

ralph_to_windows_path() {
  local unix_path="$1"

  if ! command -v cygpath >/dev/null 2>&1; then
    echo "cygpath is required to launch Windows-native Codex from bash." >&2
    return 1
  fi

  cygpath -w "$unix_path"
}

ralph_resolve_provider() {
  local requested_provider="${1:-${RALPH_LLM:-$RALPH_DEFAULT_PROVIDER}}"

  case "$requested_provider" in
    claude)
      if ! command -v claude >/dev/null 2>&1; then
        echo "LLM provider 'claude' is not installed or not on PATH." >&2
        return 1
      fi
      ;;
    codex)
      if ! command -v powershell.exe >/dev/null 2>&1; then
        echo "powershell.exe is required to launch the Codex provider from bash." >&2
        return 1
      fi
      if ! command -v cygpath >/dev/null 2>&1; then
        echo "cygpath is required to launch the Codex provider from bash." >&2
        return 1
      fi
      if [ ! -f "$RALPH_CODEX_WRAPPER" ]; then
        echo "Codex wrapper not found: $RALPH_CODEX_WRAPPER" >&2
        return 1
      fi
      ;;
    *)
      echo "Unsupported LLM provider: $requested_provider" >&2
      echo "Supported providers: claude, codex" >&2
      return 1
      ;;
  esac

  printf '%s\n' "$requested_provider"
}

ralph_print_provider_help() {
  cat <<'EOF'
LLM providers:
  claude  Uses Claude Code CLI. This remains the default provider.
  codex   Uses the Windows Codex CLI through a PowerShell wrapper from bash.

Provider selection:
  --llm claude
  --llm codex

Environment override:
  RALPH_LLM=claude
  RALPH_LLM=codex
  RALPH_CODEX_EXE=C:\full\path\to\codex.exe
EOF
}

ralph_run_headless() {
  local provider="$1"
  local prompt="$2"

  case "$provider" in
    claude)
      printf '%s' "$prompt" | claude -p --allowedTools "$RALPH_CLAUDE_ALLOWED_TOOLS"
      ;;
    codex)
      local wrapper_path_windows prompt_path_windows project_dir_windows
      wrapper_path_windows="$(ralph_to_windows_path "$RALPH_CODEX_WRAPPER")"
      prompt_path_windows="$(ralph_to_windows_path "$RALPH_PROMPT_FILE")"
      project_dir_windows="$(ralph_to_windows_path "$RALPH_PROJECT_DIR")"
      powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File "$wrapper_path_windows" \
        -Mode Headless \
        -PromptFile "$prompt_path_windows" \
        -WorkingDirectory "$project_dir_windows"
      ;;
  esac
}

ralph_run_interactive() {
  local provider="$1"
  local prompt="$2"

  case "$provider" in
    claude)
      claude "$prompt"
      ;;
    codex)
      local wrapper_path_windows prompt_path_windows project_dir_windows
      wrapper_path_windows="$(ralph_to_windows_path "$RALPH_CODEX_WRAPPER")"
      prompt_path_windows="$(ralph_to_windows_path "$RALPH_PROMPT_FILE")"
      project_dir_windows="$(ralph_to_windows_path "$RALPH_PROJECT_DIR")"
      powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File "$wrapper_path_windows" \
        -Mode Interactive \
        -PromptFile "$prompt_path_windows" \
        -WorkingDirectory "$project_dir_windows"
      ;;
  esac
}
