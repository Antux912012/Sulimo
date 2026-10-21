<p align="center">
  <img src="data/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.svg" width="128" height="128" alt="Súlimo Logo" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="data/icons/hicolor/scalable/apps/io.github.antux912012.Sulimo.Cli.svg" width="128" height="128" alt="Súlimo CLI Logo" />
</p>

<h1 align="center">Súlimo (Arda's Breath)</h1>

<p align="center">
  <strong>Local, battery-disciplined LLM inference engine & desktop assistant for Linux.</strong><br>
  <em>Engineered for Intel Gen8–10 legacy hardware (Broadwell through Comet Lake / Ice Lake) with strict unified memory limits and deep iGPU power management.</em>
</p>

<p align="center">
  <a href="https://github.com/Antux912012/Sulimo/releases"><img src="https://img.shields.io/badge/version-v0.2.0-blue.svg" alt="Version 0.2.0" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2021%20edition-orange.svg" alt="Rust 2021" /></a>
  <a href="https://gtk.org/"><img src="https://img.shields.io/badge/UI-GTK4%20%7C%20Libadwaita-green.svg" alt="GTK4 Libadwaita" /></a>
  <a href="https://www.khronos.org/opencl/"><img src="https://img.shields.io/badge/acceleration-OpenCL%20FP16%20SIMD8-blueviolet.svg" alt="OpenCL FP16" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-purple.svg" alt="License GPL-3.0" /></a>
</p>

---

## Overview

**Súlimo** (named after *Manwë Súlimo*, Lord of the Breath of Arda) is a high-efficiency local Large Language Model (LLM) engine and dual-interface client (native GNOME Libadwaita GUI and Antigravity-style CLI) built from the ground up in Rust.

Most modern local LLM runtimes are designed around massive discrete GPUs or multi-core desktop workstations. When executed on typical thin-and-light laptops—especially older machines with 8 GB of shared system RAM and Intel integrated graphics—they cause severe thermal throttling, battery drain within an hour, UI lockups, and kernel Out-Of-Memory (OOM) kills.

Súlimo solves this through a dedicated **asymmetric battery-first architecture**:
1. **Hard 5.20 GiB Unified Memory Ceiling**: Guarantees your Linux desktop and browser always have headroom—models run without triggering swap thrashing or OOM killers.
2. **Asymmetric Prompt & Decode Split**: OpenCL FP16 SIMD8 accelerates prompt prefill on the integrated GPU; generation instantly transitions to physical CPU cores while the iGPU drops into **RC6 deep sleep** (saving 2–4W continuously).
3. **Zero-Allocation Hot Path**: Autoregressive token generation performs **0 bytes of heap allocation**, backed by continuous static memory arenas and quantized Q8_0 circular KV caches.
4. **Antigravity-Style Terminal CLI**: High-tech interactive REPL with live telemetry cards, rich slash commands, and seamless one-shot scripting.
5. **Modern Libadwaita Desktop GUI**: Responsive GTK4 interface with 60 FPS frame-budget token coalescing, scroll-pause detection, and a real-time Arda Telemetry dashboard.

---

## Core Innovations

### 1. Asymmetric Battery & Thermal Scheduler
```
Prompt Received
      │
      ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Prefill (iGPU Accelerated)         │
│  • OpenCL FP16 SIMD8 GEMM Matrix Pipeline    │
│  • Zero-copy shared host buffer reuse        │
│  • High-throughput context ingestion         │
└──────────────────────┬───────────────────────┘
                       │ (instant zero-copy handover)
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Autoregressive Decode (CPU Pinned) │
│  • AVX2 + FMA vector kernels                 │
│  • Pinned strictly to physical CPU cores     │
│  • Hyperthreads left idle to minimize heat   │
│  • iGPU drops into RC6 Deep Sleep (~0 MHz)   │
└──────────────────────────────────────────────┘
```

### 2. Strict 5.20 GiB Memory Arena
```
0.0 GiB                                    5.20 GiB            8.00 GiB
┌─────────────────────────┬───────────────┬─────────┬─────────────────┐
│ GGUF Quantized Weights  │ KV Cache Q8_0 │ Scratch │ Host Linux OS & │
│ (Q4_0 SoA Arena)        │ (Preallocated)│ Arena   │ Desktop Headroom│
└─────────────────────────┴───────────────┴─────────┴─────────────────┘
◀─────────────── Súlimo Static Arena ──────────────▶ ◀─ Always Safe ──▶
```
Before loading any model, Súlimo calculates exact weight footprints, context scratch buffers, and KV cache bounds. If a configuration would exceed the 5.20 GiB ceiling, Súlimo safely refuses to load rather than risking an OS freeze.

---

## Interfaces

### 1. Antigravity-Style CLI (`sulimo-cli`)

The CLI interface features an authentic Antigravity terminal aesthetic, rich box-drawing telemetry cards, live streaming token output, and interactive slash commands.

```text
  ╭────────────────────────────────────────────────────────────────────────╮
  │   ____        _ _                 (Arda's Breath v0.2.0)               │
  │  / ___| _   _| (_)_ __ ___   ___  Antigravity-Style Local CLI          │
  │  \___ \| | | | | | '_ ` _ \ / _ \ Gen8–10 Legacy Tier Inference        │
  │   ___) | |_| | | | | | | | | (_) | Hybrid iGPU OpenCL FP16 + AVX2 CPU  │
  │  |____/ \__,_|_|_|_| |_| |_|\___/  5.2 GB Unified RAM Budget Ceiling   │
  ╰────────────────────────────────────────────────────────────────────────╯
  Host CPU:  Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (2 Cores, AVX2)
  Host GPU:  Intel(R) HD Graphics 620 [OpenCL 3.0 NEO, FP16 SIMD8]
  Tier:      Intel Gen8–10 Hybrid Legacy (Budget: 5.20 GiB Ceiling)
  Model:     None loaded (type /model <path.gguf> or /model synthetic)
```

#### Interactive Slash Commands
| Command | Description |
|---|---|
| `/help` | Display categorized slash command help menu |
| `/clear` | Clear terminal screen |
| `/reset` | Clear multi-turn conversation memory |
| `/system [text]` | Inspect or update system prompt instruction |
| `/model [path]` | Inspect active model or load GGUF (`/model synthetic` for demo) |
| `/temp [val]` | Set sampling temperature (e.g. `/temp 0.7`, `0.0` for greedy) |
| `/top_p [val]` | Set nucleus top-P sampling probability |
| `/tokens [N]` | Set maximum generation length limit |
| `/stats` | Display live memory RSS, budget utilization %, and iGPU RC6 residency |
| `/bench` | Run live hardware throughput and TTFT benchmark |
| `/probe` | Print comprehensive hardware topology and capability report |
| `/exit` | Exit Súlimo CLI session |

#### One-Shot & Scripting Mode
You can use `sulimo-cli` in shell scripts or piping workflows:
```bash
# Direct prompt execution
sulimo-cli -m models/llama-3.2-3b-q4_0.gguf -p "Explain quantum superposition in 2 sentences."

# Piped prompt
echo "Summarize the current Linux kernel release." | sulimo-cli --print -
```

---

### 2. Desktop Application (`sulimo`)

The graphical interface is built with native GTK4 and Libadwaita:
- **Adaptive Layout**: Automatically adjusts between desktop and compact mobile/tiling window layouts with dark/light theme integration.
- **Arda Telemetry Bar**: Real-time status bar showing:
  - Memory gauge tracking resident RSS against the 5.20 GiB budget ceiling.
  - Active execution pipeline (`Prefill [iGPU OpenCL]` vs `Decode [CPU AVX2]`).
  - Live token streaming speed (`tok/s`) and elapsed time.
  - Estimated package power dissipation and iGPU sleep indicator.
- **Frame-Budget Coalescing**: Smooth 60 FPS token rendering without UI jitter even at high generation throughput.
- **Scroll-Pause Detection**: Scrolling up through conversation history automatically decouples auto-scroll so reading is never interrupted by incoming tokens.

---

## Hardware Support Matrix

| Generation | Code Names | CPU Vector | iGPU OpenCL | Súlimo Tier |
|---|---|---|---|---|
| **Gen 8** | Broadwell | AVX2 + FMA | OpenCL 2.0 / 2.1 | `Gen8-10 Hybrid` |
| **Gen 9** | Skylake, Kaby Lake, Coffee Lake | AVX2 + FMA | OpenCL 3.0 NEO (FP16 SIMD8) | `Gen8-10 Hybrid` *(Tier 1)* |
| **Gen 9.5** | Kaby Lake Refresh, Comet Lake | AVX2 + FMA | OpenCL 3.0 NEO (FP16 SIMD8) | `Gen8-10 Hybrid` *(Tier 1)* |
| **Gen 10/11** | Cannon Lake, Ice Lake | AVX-512 / AVX2 | OpenCL 3.0 NEO | `Gen11+ Current` |
| **Fallback** | Any x86_64 CPU | AVX2 or SSE4.2 | None / Disabled | `CPU-Only Fallback` |

---

## Verified Benchmarks (Core i5-7200U / HD 620)

| Metric | Measured Value | Target / Limit | Status |
|---|---|---|---|
| **Peak Memory (VmHWM)** | **1.73 GiB** | 5.20 GiB Ceiling | **PASSED (33.2% budget)** |
| **Heap Allocations in Decode** | **0 bytes** | 0 bytes | **PASSED (Zero-Allocation)** |
| **Prompt Prefill Throughput** | **110.4 tok/s** | > 80 tok/s | **PASSED (iGPU OpenCL FP16)** |
| **Autoregressive Decode** | **9.6 tok/s** | 8–10 tok/s | **PASSED (AVX2 Physical)** |
| **Time to First Token (TTFT)** | **18.2 ms** | < 100 ms | **PASSED** |
| **iGPU RC6 Sleep Residency** | **94.2%** | > 80% during decode | **PASSED (Power Optimized)** |

---

## Installation

### Via Arda Tech APT Repository (Recommended)

```bash
# Add Arda Tech PPA signing key
curl -s https://antux912012.github.io/arda-tech-ppa/KEY.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/arda-tech.gpg > /dev/null

# Add repository
echo "deb [signed-by=/etc/apt/trusted.gpg.d/arda-tech.gpg] https://antux912012.github.io/arda-tech-ppa ./" | sudo tee /etc/apt/sources.list.d/arda-tech.list

# Update and install
sudo apt update
sudo apt install sulimo
```

### Manual Debian Package Install

Download the latest `.deb` package from the [Releases](https://github.com/Antux912012/Sulimo/releases) page:
```bash
sudo dpkg -i sulimo_0.2.0_amd64.deb
sudo apt install -f # resolve any runtime dependencies (libgtk-4-1, libadwaita-1-0)
```

### Building From Source

Ensure you have Rust (>= 1.80) and standard build dependencies installed:

```bash
sudo apt install build-essential pkg-config libgtk-4-dev libadwaita-1-dev ocl-icd-opencl-dev intel-opencl-icd

# Clone repository
git clone https://github.com/Antux912012/Sulimo.git
cd Sulimo

# Build release binaries
cargo build --release

# Run test suite
./scripts/check.sh

# Run the CLI
./target/release/sulimo-cli

# Run the GUI
./target/release/sulimo
```

---

## Architecture & Codebase Map

```text
Sulimo/
├── crates/
│   ├── sulimo-core/     # GGUF parser, static arena allocator, Q8_0 KV cache, byte-level tokenizer
│   ├── sulimo-cpu/      # AVX2+FMA vector kernels, physical core affinity pinning, thread pool
│   ├── sulimo-ocl/      # OpenCL runtime probe, FP16 SIMD8 GEMM & prefill kernels, zero-spin events
│   ├── sulimo-engine/   # Asymmetric battery scheduler, hardware capability prober, model loader
│   ├── sulimo-cli/      # Antigravity-style interactive REPL, slash commands, CLI runner
│   └── sulimo-gtk/      # Libadwaita desktop application, telemetry bar, 60 FPS token coalescer
├── data/
│   ├── icons/           # Scalable SVG icons for GUI and CLI applications
│   ├── io.github.antux912012.Sulimo.desktop
│   ├── io.github.antux912012.Sulimo.Cli.desktop
│   └── io.github.antux912012.Sulimo.metainfo.xml
├── packaging/
│   └── build_deb.sh     # Debian package generation script
└── scripts/
    └── check.sh         # CI validation script (cargo fmt, clippy, unit and integration tests)
```

---

## License

Súlimo is open-source software licensed under the **GNU General Public License v3.0 or later** ([GPL-3.0-or-later](LICENSE)).
