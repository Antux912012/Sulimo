#!/usr/bin/env bash
# Súlimo continuous verification script (fmt, clippy, tests).
set -euo pipefail

if [ -d "${HOME}/.local/share/sulimo/drivers/lib" ]; then
    export LD_LIBRARY_PATH="${HOME}/.local/share/sulimo/drivers/lib:${LD_LIBRARY_PATH:-}"
    export OCL_ICD_VENDORS="${HOME}/.local/share/sulimo/drivers/icd"
fi

echo "==> Running cargo fmt check..."
cargo fmt --all -- --check

echo "==> Running cargo clippy with -D warnings..."
cargo clippy --workspace --all-targets -- -D warnings

echo "==> Running cargo test workspace..."
cargo test --workspace

echo "==> All Súlimo workspace checks passed successfully!"
