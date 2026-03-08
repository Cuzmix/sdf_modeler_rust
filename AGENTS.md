# SDF Modeler Agent Rules

This repository enforces the workflow and quality rules in `CLAUDE.md`.

## Mandatory Validation Before Commit
Every commit must pass, in this order:
1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. Manual verification for visual/behavioral changes

Do not commit if any step fails.

## Scope and Architecture
- Keep changes modular and discoverable. Avoid monolithic files.
- Prefer explicit, readable naming over short naming.
- Follow Rust idioms and keep code junior-friendly.
- Do not add placeholder/stub implementations.

## Performance
- Performance must not regress.
- Prefer focused hand-rolled implementations over unnecessary dependencies.

## Ralph / Task Workflow
- One logical change per commit.
- Keep commits focused with descriptive messages.
- When using PRD/progress workflow, commit code + PRD + progress updates together.

## Source of Truth
For full details and project architecture context, read:
- `CLAUDE.md`
- `docs/architecture.md`
