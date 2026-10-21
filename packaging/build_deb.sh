#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VERSION=$(grep -m1 '^version' "${ROOT_DIR}/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
STAGE_DIR="${ROOT_DIR}/target/deb_stage"

echo "==> Building Sulimo (Rust Engine + UI) v${VERSION}..."
cargo build --release --locked

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/DEBIAN" \
         "${STAGE_DIR}/usr/bin" \
         "${STAGE_DIR}/usr/share/applications" \
         "${STAGE_DIR}/usr/share/metainfo" \
         "${STAGE_DIR}/usr/share/icons/hicolor/scalable/apps" \
         "${STAGE_DIR}/usr/share/doc/sulimo"

chmod 755 "${STAGE_DIR}/DEBIAN"

# Install binaries
install -m 755 "${ROOT_DIR}/target/release/sulimo" "${STAGE_DIR}/usr/bin/sulimo"
install -m 755 "${ROOT_DIR}/target/release/sulimo-cli" "${STAGE_DIR}/usr/bin/sulimo-cli"

# Compatibility symlink: arda-probe -> sulimo-cli
ln -sf sulimo-cli "${STAGE_DIR}/usr/bin/arda-probe"

# FreeDesktop Desktop, Metainfo, Icons
install -m 644 "${ROOT_DIR}/data/io.github.antux912012.Sulimo.desktop" "${STAGE_DIR}/usr/share/applications/io.github.antux912012.Sulimo.desktop"
install -m 644 "${ROOT_DIR}/data/io.github.antux912012.Sulimo.Cli.desktop" "${STAGE_DIR}/usr/share/applications/io.github.antux912012.Sulimo.Cli.desktop"
install -m 644 "${ROOT_DIR}/data/io.github.antux912012.Sulimo.metainfo.xml" "${STAGE_DIR}/usr/share/metainfo/io.github.antux912012.Sulimo.metainfo.xml"
install -m 644 "${ROOT_DIR}/data/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.svg" "${STAGE_DIR}/usr/share/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.svg"
install -m 644 "${ROOT_DIR}/data/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.Cli.svg" "${STAGE_DIR}/usr/share/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.Cli.svg"

# Documentation
install -m 644 "${ROOT_DIR}/README.md" "${STAGE_DIR}/usr/share/doc/sulimo/README.md"

cat <<EOF > "${STAGE_DIR}/DEBIAN/control"
Package: sulimo
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: amd64
Depends: libc6, libgtk-4-1, libadwaita-1-0
Recommends: intel-opencl-icd | mesa-opencl-icd, ocl-icd-libopencl1
Provides: arda-probe
Maintainer: Antonio <antux912012@users.noreply.github.com>
Homepage: https://github.com/Antux912012/Sulimo
Description: Sulimo (Arda's Breath) - Local LLM Engine & Hardware Telemetry
 High-performance, battery-disciplined local LLM inference engine optimized
 for Intel Gen8–10 legacy hardware (Broadwell through Comet Lake / Ice Lake).
 Features zero-copy GGUF static arena memory, asymmetric OpenCL FP16 SIMD8
 prompt prefill, AVX2 physical-core decode with iGPU RC6 deep sleep, and
 a responsive Libadwaita/GTK4 desktop interface.
EOF

DEB_PATH="${ROOT_DIR}/sulimo_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "${STAGE_DIR}" "${DEB_PATH}"
rm -rf "${STAGE_DIR}"
echo "==> Built ${DEB_PATH}"
