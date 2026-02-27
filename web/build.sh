#!/usr/bin/env bash
set -euo pipefail

# Build the WASM binary
echo "Building WASM..."
cargo build --release --target wasm32-unknown-unknown --lib

# Run wasm-bindgen to generate JS glue
echo "Running wasm-bindgen..."
wasm-bindgen \
    --out-dir web/pkg \
    --target web \
    --no-typescript \
    target/wasm32-unknown-unknown/release/sdf_modeler.wasm

# Optional: optimize with wasm-opt if available
if command -v wasm-opt &> /dev/null; then
    echo "Optimizing with wasm-opt..."
    wasm-opt -Oz web/pkg/sdf_modeler_bg.wasm -o web/pkg/sdf_modeler_bg.wasm
fi

echo ""
echo "Build complete! Serve the web/ directory with a local HTTP server:"
echo "  python -m http.server -d web 8080"
echo "  # Then open http://localhost:8080 in a WebGPU-capable browser (Chrome 113+)"
