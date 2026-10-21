#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION=$(grep -m1 '^version' "${ROOT_DIR}/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
REPO_DIR="${ROOT_DIR}/target/flatpak-repo"
BUILD_DIR="${ROOT_DIR}/target/flatpak-build"
BUNDLE_PATH="${ROOT_DIR}/sulimo_${VERSION}.flatpak"

echo "==> 1. Ensuring release binaries are up-to-date..."
cargo build --release --locked --bin sulimo --bin sulimo-cli

echo "==> 2. Building Sulimo Flatpak v${VERSION} in OSTree repository..."
mkdir -p "${REPO_DIR}" "${BUILD_DIR}"

flatpak-builder \
  --force-clean \
  --disable-rofiles-fuse \
  --repo="${REPO_DIR}" \
  "${BUILD_DIR}" \
  "${SCRIPT_DIR}/io.github.antux912012.Sulimo.json"

echo "==> Exporting Flatpak single-file bundle..."
flatpak build-bundle \
  --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo \
  "${REPO_DIR}" \
  "${BUNDLE_PATH}" \
  io.github.antux912012.Sulimo \
  master

echo "==> Flatpak bundle generated successfully at ${BUNDLE_PATH}"
