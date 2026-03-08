param(
    [int]$BenchmarkSeconds = 12
)

$ErrorActionPreference = "Stop"

cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_a_slint_owned -- --benchmark-seconds $BenchmarkSeconds
cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_b_winit_owned -- --benchmark-seconds $BenchmarkSeconds