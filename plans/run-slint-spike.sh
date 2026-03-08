#!/usr/bin/env bash
set -euo pipefail

SECONDS="${1:-12}"

cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_a_slint_owned -- --benchmark-seconds "$SECONDS"
cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_b_winit_owned -- --benchmark-seconds "$SECONDS"