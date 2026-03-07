---
name: verify
description: Run the mandatory verification loop — cargo check, clippy, test, and build. Use before committing changes, when asked to verify, validate, or check the build, or after completing a feature.
allowed-tools: Bash, Read
---

# Verification Loop

Run these 4 commands **sequentially**. Stop on the first failure, fix the issue, then restart from step 1.

## Steps

1. **Type check** (fastest feedback):
   ```bash
   cargo check
   ```

2. **Lint** (warnings are errors):
   ```bash
   cargo clippy -- -D warnings
   ```

3. **Tests** (all must pass):
   ```bash
   cargo test
   ```

4. **Full build**:
   ```bash
   cargo build
   ```

## On Failure

- Report the exact error with file path and line number
- Fix the root cause (do not suppress warnings with `#[allow(...)]` unless the lint is genuinely a false positive)
- Restart from step 1 after fixing

## On Success

Report: "All 4 verification steps passed" with the test count (e.g., "380 passed").

## Rules

- Never skip steps or run them out of order
- Never use `--no-verify` on git commits
- If clippy suggests a fix, apply it unless it would degrade readability or performance
- This loop is MANDATORY before every commit per project rules (CLAUDE.md)
