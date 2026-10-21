#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VERSION=$(grep -m1 "project(arda_breath" "${ROOT_DIR}/CMakeLists.txt" | sed -E 's/.*VERSION ([0-9.]+).*/\1/')
STAGE_DIR="${ROOT_DIR}/build/deb_stage"

echo "==> Building Sulimo (arda-probe) v${VERSION}..."
cmake -B "${ROOT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/build" --parallel

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/DEBIAN" "${STAGE_DIR}/usr/bin" "${STAGE_DIR}/usr/share/doc/sulimo"
install -m 755 "${ROOT_DIR}/build/tools/arda-probe/arda-probe" "${STAGE_DIR}/usr/bin/arda-probe"
ln -sf arda-probe "${STAGE_DIR}/usr/bin/sulimo"
install -m 644 "${ROOT_DIR}/README.md" "${STAGE_DIR}/usr/share/doc/sulimo/README.md"

cat <<EOF > "${STAGE_DIR}/DEBIAN/control"
Package: sulimo
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: amd64
Depends: libc6
Provides: arda-probe
Maintainer: Antonio <antux912012@users.noreply.github.com>
Homepage: https://github.com/Antux912012/arda-tech
Description: Sulimo (Arda's Breath) Hardware Capability Probe & Runtime Engine
 High-performance C11 hardware capability probe (arda-probe) and runtime
 optimization policy selector for CPU microarchitecture, memory topology,
 Vulkan, OpenCL, and SYCL compute backends.
EOF

DEB_PATH="${ROOT_DIR}/sulimo_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "${STAGE_DIR}" "${DEB_PATH}"
rm -rf "${STAGE_DIR}"
echo "==> Built ${DEB_PATH}"
