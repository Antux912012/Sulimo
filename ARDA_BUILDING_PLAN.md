# Project: Arda's Breath - Local AI Inference Engine & Desktop Client

## 1. Project Context & Philosophy
**Arda's Breath** is a highly resource-conscious local AI execution environment designed to make small-to-medium AI models accessible on older and constrained hardware (8GB–32GB RAM). 

**Inspirations:**
*   **`llama.cpp`**: The foundational inference backend.
*   **`ds4` (DwarfStar 4)**: Inspiration for low-overhead, specialized, and highly optimized inference principles.
*   **`colibri`**: Inspiration for extreme memory-footprint optimization, pure C implementation, and disk-streaming for Mixture-of-Experts (MoE) models.

**Core Philosophy:** Extreme resource care. The software must not just run on legacy hardware; it must run *efficiently*, treating RAM, CPU cycles, and I/O bandwidth with strict discipline.

## 2. Architecture Overview
The project utilizes a unified frontend with two distinct, profile-optimized backend cores.

*   **Frontend: "Arda's Breath"**
    *   A single, unified GTK4 / Libadwaita desktop application (built in C to ensure an ultra-lightweight <40MB memory footprint).
    *   Distributed as a Flatpak.
    *   UI/UX inspired by Google Antigravity: clean, chat-first, hiding unnecessary complexity, with a "plan-first" visual workflow layout.
*   **Core 1: "Arda Classic"**
    *   **Target:** Intel 4th to 8th Gen (Haswell to Coffee Lake). 8GB-16GB DDR3/DDR4. SATA SSDs/HDDs. Intel HD 4000-600 series iGPUs.
    *   **Strategy:** AVX2 is the primary workhorse. GPU (OpenCL/Vulkan) is strictly limited to prompt prefill to avoid DDR3 bandwidth saturation; token generation is forced to CPU. Disk streaming uses large sequential chunks (4MB+) to mitigate SATA latency.
*   **Core 2: "Arda Current"**
    *   **Target:** Intel 9th Gen to Current (Ice Lake to Meteor Lake+). 16GB-32GB+ DDR5/LPDDR5. NVMe SSDs. Iris Xe / Intel Arc.
    *   **Strategy:** AVX-VNNI / AVX-512 for CPU. First-class support for Intel SYCL (oneAPI) and Vulkan for full GPU offload (prefill and decode). Disk streaming uses fine-grained chunking (64KB-256KB) optimized for NVMe random reads.

## 3. Execution Plan for AI Builder

**Instructions for the AI:** You are an expert systems architect and C/GTK4 developer. Execute this project phase by phase. **Do not proceed to the next phase until I have reviewed and approved the code and architecture of the current phase.** Provide complete, compilable code, configuration files, and build instructions for each step.

### Phase 0: Repository Structure & Dual-Core Foundation
1.  **Monorepo Setup:** Initialize a repository with `/core-classic`, `/core-current`, `/frontend`, and `/shared` (for common headers and IPC definitions).
2.  **CMake Presets:** Create `CMakePresets.json` with two distinct build profiles:
    *   `arda-classic`: `-march=haswell`, AVX2 enabled, AVX-512 disabled, minimal OpenCL/Vulkan fallbacks.
    *   `arda-current`: `-march=x86-64-v3` (or v4), AVX-VNNI enabled, full SYCL/Vulkan backend enabled.
3.  **Hardware Probe:** Write a lightweight C utility in `/shared` that reads `/proc/cpuinfo` and `/proc/meminfo` to detect the CPU generation, AVX capabilities, and total RAM. This will dictate which core is loaded at runtime.

### Phase 1: Core Engine Integration & Profile Divergence
1.  **Backend Base:** Integrate `llama.cpp` as a Git submodule in both core directories.
2.  **Classic Core Configuration:** 
    *   Configure CMake to aggressively optimize AVX2 kernels.
    *   Implement logic to default to CPU-only inference for token generation (decode), exposing Vulkan/OpenCL only for prompt prefill.
3.  **Current Core Configuration:**
    *   Enable Intel SYCL and Vulkan backends.
    *   Configure for full GPU offload (layers offloaded to VRAM/shared memory for both prefill and decode).
4.  **Deliverable:** Both cores must successfully compile and run a basic GGUF model via a temporary test CLI.

### Phase 2: Advanced Memory & I/O Optimization (The "Colibri" Influence)
1.  **Hybrid Memory Management:** Implement `mmap` for the entire model, but use `mlock()` or `madvise(MADV_WILLNEED)` to pin critical layers (attention weights, RMS norms) in physical RAM.
2.  **Profile-Specific I/O:**
    *   *Classic:* Implement a disk-streaming buffer that reads model weights in large, sequential blocks (4MB+) to mitigate SATA random-read latency.
    *   *Current:* Implement fine-grained, page-level `mmap` access optimized for NVMe random reads (64KB chunks) for rapid MoE expert swapping.
3.  **Memory Reclamation:** When the engine is idle, explicitly call `madvise(..., MADV_DONTNEED)` on model memory pages to return RAM to the OS.
4.  **Deliverable:** Memory profiling scripts and updated core code demonstrating sub-8GB RAM usage for a 7B/8B model on the Classic profile.

### Phase 3: CLI Service Daemon & Zero-Copy IPC
1.  **Daemon Creation:** Build `ardad` (Arda's Breath Daemon), a headless CLI service that loads the appropriate core (Classic or Current) based on the Phase 0 hardware probe.
2.  **Zero-Copy IPC:** Implement Inter-Process Communication between the daemon and the frontend using **POSIX Shared Memory** (`memfd_create` or `shm_open`). This prevents duplicating large payloads (chat history, model metadata) in RAM.
3.  **Local API:** Implement a lightweight local WebSocket/REST API exposing:
    *   `/api/v1/models`: List, download, delete.
    *   `/api/v1/chat`: Streaming token generation.
    *   `/api/v1/system`: Real-time telemetry (RAM, VRAM, CPU load, active profile).
4.  **Deliverable:** A fully functional background daemon with API documentation and IPC test scripts.

### Phase 4: Model Management System (LM Studio Style)
1.  **Model Registry:** Integrate a lightweight Hugging Face API client to search for GGUF repositories.
2.  **Hardware-Aware Curation:** Implement an "Arda Certified" filter. If the Classic profile is active, hide or warn against models >4GB. Recommend Q3_K_M / Q4_K_M. If Current is active, allow Q5_K_M / Q6_K and larger context windows.
3.  **Download Manager:** Implement resumable, chunked downloading with progress tracking exposed via the WebSocket API.
4.  **Deliverable:** CLI commands and API endpoints capable of discovering, downloading, and verifying GGUF models autonomously.

### Phase 5: GTK4 / Libadwaita Frontend ("Arda's Breath")
1.  **Technology Stack:** Build the frontend in **C** using the GTK4 and Libadwaita C APIs to guarantee an ultra-lightweight memory footprint (<40MB).
2.  **UI/UX Design (Antigravity Inspired):**
    *   Clean, chat-first interface.
    *   Collapsible sidebar for model selection, download management, and system resource monitoring.
    *   "Plan-first" visual workflow panel for system prompts/agent instructions.
3.  **Adaptive Rendering:** 
    *   Load a dynamic CSS file based on the daemon's reported profile.
    *   *Classic Profile:* Inject CSS to disable Libadwaita animations (`all 0ms`), remove backdrop blurs, and simplify widget rendering to preserve iGPU cycles.
    *   *Current Profile:* Enable default Libadwaita styling for fluid animations.
4.  **Deliverable:** A fully interactive, visually polished, and highly optimized GTK4 desktop application connected to the daemon.

### Phase 6: Flatpak Packaging & Final Tuning
1.  **Flatpak Manifest:** Create `org.arda.breath.yml`.
    *   Bundle the daemon and frontend.
    *   Declare GPU extensions (`org.freedesktop.Platform.VulkanLayer`, OpenCL ICD).
    *   Configure sandbox permissions: `--filesystem=xdg-download/ArdaBreath` for models, `--device=dri` for GPU, and shared memory access.
2.  **Profile-Aware Wrapper:** The Flatpak entry script must intelligently pass environment variables (e.g., `GGML_VULKAN=1` for Current, `GGML_VULKAN=0` for Classic) based on the runtime hardware detection.
3.  **Deliverable:** A buildable Flatpak package, a comprehensive `README.md`, and a testing script that simulates resource constraints (using `cgroups` to cap RAM at 8GB) to verify stability.

---

## 4. Strict Rules for the AI Builder
1.  **No Hallucinated Dependencies:** Use standard C libraries, GTK4, Libadwaita, and `llama.cpp`. Do not introduce heavy external frameworks unless explicitly requested.
2.  **Resource Discipline:** Every line of code must be evaluated for its memory and CPU footprint. Avoid unnecessary allocations.
3.  **Step-by-Step Execution:** Output the code for **Phase 0** only. Stop and wait for my prompt to proceed to Phase 1. 
4.  **Code Quality:** Ensure all C code is modular, well-commented, and follows modern C standards (C11/C17). Use CMake for all build configurations.