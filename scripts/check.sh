#!/usr/bin/env bash
# Súlimo continuous verification script (fmt, clippy, tests).
set -euo pipefail

echo "==> Running cargo fmt check..."
cargo fmt --all -- --check

echo "==> Running cargo clippy with -D warnings..."
cargo clippy --workspace --all-targets -- -D warnings

echo "==> Running cargo test workspace..."
cargo test --workspace

echo "==> All Súlimo workspace checks passed successfully!"
