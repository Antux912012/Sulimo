# Arda's Breath

Local LLM inference engine and lightweight desktop client designed for resource-conscious execution across diverse Linux hardware.

## Project Status: Phase 0 Complete

Phase 0 establishes the engineering foundation:
- Portable C11/C17 baseline build system with CMake and presets.
- Shared hardware capability probe library (`arda_shared`) and CLI tool (`arda-probe`).
- Independent, decoupled compile-time switches and runtime probes for CPU and GPU backends (Vulkan, OpenCL, SYCL).
- Objective capability evaluation and runtime profile selection (`arda-portable`, `arda-classic`, `arda-current`).
- Comprehensive unit test suite with mock fixtures and live hardware validation.

---

## Architecture Principles

1. **Correctness before optimization:** Stable baseline and reproducible tests before custom memory strategies.
2. **Capability detection over generation labels:** CPU instruction capabilities (AVX2, AVX-512, AVX-VNNI), RAM capacity, and GPU backends are probed directly at runtime; they are never inferred from CPU model strings.
3. **Graceful degradation and safe runtime probes:** GPU backends (Vulkan, OpenCL) are queried safely at runtime via dynamic loader checks (`dlopen`). Absence of GPU drivers or developer headers does not fail the build or cause runtime crashes.
4. **Decoupled execution profiles:** Profiles are runtime policies, not divergent inference codebases.

---

## Directory Structure

```text
.
├── CMakeLists.txt          # Root CMake build configuration
├── CMakePresets.json       # Build and test presets (portable, classic, current, etc.)
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── THIRD_PARTY_NOTICES.md  # Third-party attribution and licenses
├── core/                   # Shared inference adapter & execution policy (Phase 1+)
├── daemon/                 # Headless ardad service & local API (Phase 3)
├── frontend/               # GTK4 / Libadwaita client (Phase 5)
├── model_manager/          # Model registry and download manager (Phase 4)
├── packaging/              # Packaging manifests and Flatpak definitions (Phase 6)
├── shared/                 # Shared capability probe, profile evaluator, common headers
│   ├── include/arda/       # Public C headers (capability.h, profile.h, version.h)
│   └── src/                # Probes for OS, CPU, Memory, GPU backends, formatting
├── tools/
│   └── arda-probe/         # Command-line capability probe utility
├── tests/
│   ├── fixtures/           # Offline cpuinfo and meminfo fixtures
│   └── unit/               # Unit tests for CPU, memory, profiles, and live probe
└── benchmarks/             # Performance benchmark scripts and logs (Phase 1+)
```

---

## Building and Testing

### Prerequisites
- C11/C17 compatible compiler (GCC or Clang)
- CMake >= 3.20
- Ninja or GNU Make
- Standard C library and `libdl`

### Quick Start (Portable Baseline)

Configure and build using CMake presets:

```bash
# Configure portable baseline
cmake --preset portable

# Build all targets
cmake --build --preset portable

# Run test suite
ctest --preset portable
```

### Manual Configuration

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Running the Hardware Capability Probe

```bash
# Human-readable report (default)
./build/portable/tools/arda-probe/arda-probe

# JSON report (matches daemon API schema)
./build/portable/tools/arda-probe/arda-probe --json

# Quick sanity check
./build/portable/tools/arda-probe/arda-probe --check
```

---

## Build Presets

| Preset | Purpose | Flags / Switches |
|---|---|---|
| `portable` | Default distribution baseline | Portable C11, no microarchitecture assumptions, GPU backends OFF |
| `cpu-only` | Explicit CPU-only build | All GPU backends disabled, portable flags |
| `classic` | Haswell-Coffee Lake target | Tuned for AVX2 workhorse systems |
| `current` | Modern CPU target | Tuned for AVX-512 / AVX-VNNI systems |
| `vulkan` | Vulkan-enabled build | `ARDA_ENABLE_VULKAN=ON` |
| `opencl` | OpenCL-enabled build | `ARDA_ENABLE_OPENCL=ON` |
| `native` | Local development tuning | `-march=native` (not for distribution) |

---

## Documented Assumptions and Constraints

- **Non-privileged probing:** The capability probe reads only standard user-accessible system interfaces (`sysconf`, `sysinfo`, `/proc/cpuinfo`, `/proc/meminfo`, dynamic libraries). No elevated privileges (root/sudo) are required.
- **Robust parsing:** Both `/proc/cpuinfo` and `/proc/meminfo` parsers tolerate missing fields, unexpected key orders, or arbitrary formatting.
- **Hardware Profile Policies:**
  - `arda-portable`: Assigned when AVX2 is absent; enforces baseline CPU execution and conservative memory budgets.
  - `arda-classic`: Assigned when AVX2 is supported but AVX-512/AVX-VNNI is absent, or on systems with integrated graphics and 8–16 GB RAM.
  - `arda-current`: Assigned when AVX-512 or AVX-VNNI instructions are verified and the system has >= 16 GB RAM or a capable discrete GPU.
