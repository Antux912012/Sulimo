# Arda's Breath — Engineering Plan
## Local LLM inference engine and lightweight desktop client

**Status:** Proposed implementation plan  
**Primary coding agent:** Gemini  
**Implementation language:** C11/C17 for project-owned native code  
**Build system:** CMake  
**Initial inference backend:** `llama.cpp`  
**Primary target:** Linux desktop, including constrained 8–32 GB RAM systems  
**GPU priorities:** Vulkan first; OpenCL and Intel SYCL as optional, evidence-driven backends  
**Desktop UI:** GTK4 + Libadwaita  
**Packaging target:** Flatpak, after native builds and runtime behavior are validated

---

## 1. Mission and design principles

Arda's Breath is a resource-conscious local LLM application designed to run useful small-to-medium models on a broad range of Linux hardware, from older CPU-only machines to systems with capable integrated or discrete GPUs.

The project should prioritize:

1. **Correctness before optimization.** Establish a reliable baseline and reproducible tests before adding custom memory or kernel strategies.
2. **One inference engine, multiple execution strategies.** Do not maintain separate copies of the inference engine for older and newer machines. Use profiles as runtime policies and test configurations.
3. **Capability detection, not generation-name assumptions.** Detect CPU features, available GPU devices, driver/backend availability, and memory constraints independently.
4. **Measure every performance claim.** Never assume that a GPU, smaller I/O chunk, pinned page, or special build is faster without a reproducible benchmark.
5. **Resource budgets are explicit.** Account for model weights, KV cache, compute buffers, GPU allocations, staging memory, application overhead, and OS headroom.
6. **Use established infrastructure first.** Treat `llama.cpp` as the initial inference implementation. Do not fork or replace its loader, scheduler, or kernels without measured evidence that a custom implementation is necessary.
7. **Graceful fallback.** If a GPU backend is unavailable or fails to initialize, the application must explain why and offer a CPU fallback when possible.
8. **Incremental delivery.** Implement only the current phase. Stop for human review and explicit approval before beginning the next phase.

## 2. Product scope

### In scope

- Local GGUF model inference through `llama.cpp`.
- CPU inference and optional GPU acceleration.
- Runtime hardware/backend capability report.
- Model loading, unloading, cancellation, and resource lifecycle.
- A headless local daemon and documented local API.
- Model discovery and resumable downloads with integrity checks.
- GTK4/Libadwaita desktop chat interface.
- Resource telemetry and reproducible performance benchmarks.
- Linux packaging, including Flatpak once native behavior is stable.

### Out of scope for the initial release

- Writing a new inference engine from scratch.
- Maintaining two divergent copies of `llama.cpp`.
- Custom tensor kernels before profiling establishes a need.
- Automatic online execution of downloaded code or model-repository scripts.
- Cloud inference, accounts, telemetry uploads, or mandatory external services.
- Promising that every model will fit or run quickly on every machine.

## 3. Proposed architecture

```text
GTK4 / Libadwaita client
          |
          | versioned local API; streaming responses
          v
     arda daemon (ardad)
          |
          +-- request/session lifecycle
          +-- model registry and download manager
          +-- resource budget and telemetry
          +-- backend/capability report
          |
          v
   shared inference adapter
          |
          v
       llama.cpp
          |
          +-- CPU backend
          +-- Vulkan backend (first GPU backend to evaluate)
          +-- OpenCL backend (optional; benchmark and support dependent)
          +-- Intel SYCL backend (optional; benchmark and support dependent)
```

The diagram describes logical responsibilities, not a requirement that every component be a separate process or library on day one.

### Architecture rules

- Maintain one shared inference integration and model lifecycle.
- Keep `arda-classic` and `arda-current` as named presets or policy profiles, not separate inference codebases.
- Separate CPU capability detection from GPU/backend discovery.
- Keep GPU backends optional at build time and runtime.
- Do not assume that the existence of a backend library means a compatible GPU is available.
- Keep the UI independent from inference implementation details through a versioned API.
- Keep model files outside the application bundle.
- Do not add shared-memory IPC until profiling demonstrates a real copy/latency bottleneck.

## 4. Hardware profiles and capability detection

Profiles are starting policies, not hard-coded rules about what hardware can do.

### Classic profile

Intended for older or resource-constrained systems, such as machines with 8–16 GB RAM, older x86-64 CPUs, SATA storage, or limited/unsupported GPUs.

Default behavior:
- Prefer a portable and reliable CPU path.
- Enable supported CPU optimizations only when the build and runtime support them.
- Offer GPU acceleration only if a backend initializes and a benchmark or known-good policy indicates it is appropriate.
- Use conservative default context and memory budgets.
- Do not automatically pin model pages or introduce custom disk streaming.

### Current profile

Intended for more capable systems, such as machines with 16–32+ GB RAM, newer CPUs, NVMe storage, and supported integrated or discrete GPUs.

Default behavior:
- Evaluate Vulkan GPU offload first when available.
- Permit partial or full GPU offload according to backend support and available memory.
- Allow optional OpenCL/SYCL builds where useful.
- Use measured policies for prompt prefill and token decode; do not assume the same policy is optimal for both.
- Still preserve CPU fallback and memory limits.

### Required capability report

Implement a small, testable report containing, where detectable:

- OS and architecture.
- CPU vendor/model string and logical CPU count.
- Runtime CPU instruction capabilities relevant to compiled kernels.
- Total and currently available system memory.
- Page size.
- Relevant GPU device identifiers and memory information where the API exposes them.
- Which backends were compiled in.
- Which backends can initialize at runtime.
- Driver/runtime errors and unsupported capability reasons.
- Selected profile and the reason for that selection.

Do not infer AVX-512, AVX-VNNI, Vulkan, OpenCL, or SYCL support from a CPU generation name alone.

Linux-specific sources such as `/proc/cpuinfo` and `/proc/meminfo` may be used, but parsing must tolerate missing fields and unexpected formatting. Prefer established feature-detection mechanisms where available. Keep the probe read-only and safe on systems where a GPU driver or optional runtime is absent.

## 5. Build and portability strategy

Use CMake and C11/C17 for project-owned C code. Use C++ only where required by an upstream dependency or a backend integration that cannot reasonably be called through a stable C interface; document the reason.

### Build principles

- Start with a portable baseline build.
- Scope architecture-specific compiler flags to the relevant targets; do not accidentally make the entire application unusable on older CPUs.
- Treat AVX2 and newer instruction sets as optional build/runtime capabilities.
- Do not use `-march=native` for release binaries intended for distribution.
- Do not assume `-march=x86-64-v3` enables AVX-VNNI.
- Do not assume `-march=x86-64-v4` is safe on every modern x86-64 processor; it requires a suitable AVX-512-capable CPU.
- Prefer upstream `llama.cpp` CPU dispatch and supported CMake options rather than recreating kernel dispatch.
- Keep Vulkan, OpenCL, and SYCL configuration explicit and independently switchable.
- Pin or record the tested `llama.cpp` revision. Update it deliberately and rerun regression/benchmark tests.

### Suggested initial repository layout

```text
arda-breath/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── .gitmodules
├── core/
│   ├── CMakeLists.txt
│   ├── inference_adapter/
│   └── runtime_policy/
├── shared/
│   ├── include/arda/
│   └── src/
├── daemon/
│   ├── CMakeLists.txt
│   └── src/
├── frontend/
│   ├── CMakeLists.txt
│   └── src/
├── model_manager/
│   ├── CMakeLists.txt
│   └── src/
├── packaging/
│   └── flatpak/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── benchmarks/
    ├── README.md
    ├── scripts/
    └── results/
```

Gemini may propose a better layout during Phase 0, but it must explain the trade-off before changing this structure.

## 6. Inference and GPU execution policy

### Baseline

First get a known GGUF model running through `llama.cpp` on CPU. Record model identity, quantization, context length, thread count, build configuration, and output correctness.

### Backend rollout

1. CPU baseline.
2. Vulkan build and runtime validation.
3. OpenCL only if target-device coverage or benchmark results justify it.
4. Intel SYCL as an optional Intel-specific configuration, benchmarked against Vulkan where both are supported.

Do not bundle every backend into every build by default.

### Execution modes to benchmark

- CPU-only inference.
- GPU prompt prefill with CPU token decode, where supported.
- Partial GPU offload.
- Full GPU offload.
- Backend-specific configurations supported by the pinned `llama.cpp` revision.

The selected mode must be based on explicit settings, a known-good profile, or measured benchmark results. Do not silently change execution strategy based on unsupported assumptions.

### Required measurements

For each representative hardware/model combination, record:

- Prompt prefill throughput in tokens/second.
- Decode throughput in tokens/second.
- Time to first token.
- Peak process resident memory.
- Peak GPU memory or best available proxy.
- Model load time.
- Failures, unsupported operations, and fallback behavior.
- Power or energy measurements only if reliable tools are available.

Test at least a short prompt and a long prompt. Prefill and decode must be reported separately because their performance characteristics differ.

## 7. Memory management and model lifecycle

Use the existing `llama.cpp` model-loading and memory behavior as the baseline. Do not add a custom model loader or weight streamer in the initial implementation.

### Budget components

Estimate or measure:
- Model weights and mappings.
- KV cache for the configured context and sequence count.
- Compute and temporary buffers.
- GPU allocations and host staging buffers.
- Tokenizer and runtime allocations.
- Daemon and frontend memory.
- A safety margin for the OS and other running applications.

### Lifecycle requirements

- Load and unload models explicitly.
- Report estimated and observed memory usage where possible.
- Handle allocation failures with actionable error messages.
- Allow users to configure context size and memory limits.
- Avoid loading a second large model while the current model is still resident unless resources permit it.
- Clean up backend contexts and allocations on cancellation, errors, and shutdown.
- Keep mmap, `mlock`, `madvise`, asynchronous I/O, and custom streaming as experiments until measurements justify them.
- Do not promise that `MADV_DONTNEED` will reclaim all memory or file cache.
- Do not pin pages by default.

### Memory benchmark target

For any claim such as “runs an 7B/8B model below 8 GB RAM,” document the exact model file, quantization, context length, concurrency, backend, OS, and measurement method. Treat the result as a tested configuration, not a universal guarantee.

## 8. Daemon, API, and IPC

The daemon (`ardad`) owns inference requests and model lifecycle. The UI communicates through a versioned local API.

### Initial API surface

- `GET /api/v1/system` — capability report, selected profile, resource telemetry.
- `GET /api/v1/models` — installed models and metadata.
- `POST /api/v1/models/load` — load a selected model.
- `POST /api/v1/models/unload` — unload the active model.
- `POST /api/v1/chat` — submit a chat request with a streaming response.
- `POST /api/v1/chat/cancel` — cancel a request.
- Model discovery/download endpoints may be added in the model-manager phase.

The transport can be HTTP with streaming responses initially. Add WebSocket only if the requirements demonstrate that bidirectional messaging is needed. Document the transport choice and its limitations.

### Security and reliability requirements

- Bind to loopback by default; never expose the API to the LAN without explicit user configuration.
- Do not use permissive CORS or unauthenticated non-loopback access.
- Validate request sizes, JSON fields, context limits, model paths, and download destinations.
- Avoid shelling out with untrusted strings.
- Define behavior for client disconnects, daemon restarts, duplicate requests, and cancellation.
- Version API schemas and document error responses.
- Add shared-memory IPC only after profiling and with a documented synchronization/lifetime design.

## 9. Model registry and downloads

### Model management features

- Search supported model repositories.
- Show model name, repository, quantization, file size, license metadata where available, and source URL.
- Distinguish verified facts from estimates and repository-provided claims.
- Estimate fit based on weights, context/KV cache, backend memory, and system headroom.
- Allow users to set download location and cancel downloads.
- Support resumable downloads where the server permits it.
- Verify checksums or other trustworthy integrity metadata when available.
- Download to a temporary file and publish it as installed only after validation.
- Handle disk-full conditions, interrupted downloads, invalid metadata, and duplicate files.
- Never execute downloaded repository scripts or binaries as part of model installation.

### Compatibility labels

Use clear categories such as:
- **Tested:** this exact model/configuration passed a recorded test.
- **Likely to fit:** estimated memory requirements are within configured limits.
- **Unverified:** insufficient compatibility or performance data.
- **Unsupported:** a known incompatibility exists.

Do not classify a model as “certified” solely because its file size is below a fixed threshold.

## 10. GTK4 / Libadwaita frontend

Build a native desktop client in C using GTK4 and Libadwaita.

### Initial UX

- Chat-first interface.
- Model selection and load/unload state.
- Streaming token display.
- Stop/cancel generation.
- Clear errors when a model or backend cannot load.
- Collapsible model/download sidebar.
- System status for RAM, active backend, and inference speed.
- A “plan-first” area for system prompts or agent instructions, without implying that the application can guarantee model correctness.

### Performance rules

- Keep inference off the UI thread.
- Use asynchronous API calls and avoid blocking the GTK main loop.
- Avoid excessive polling; prefer event-driven updates or modest telemetry intervals.
- Measure actual frontend resident memory before claiming a fixed footprint such as less than 40 MB.
- Do not disable animations or simplify styling unless profiling demonstrates a meaningful benefit.
- Provide accessible controls, keyboard navigation, and readable error messages.

## 11. Flatpak and distribution

Do not start Flatpak packaging until the native daemon and frontend run reliably on the target distribution(s).

### Packaging requirements

- Build a reproducible Flatpak manifest.
- Declare only the permissions and runtime extensions actually required.
- Verify Vulkan and OpenCL ICD/runtime visibility from inside the sandbox.
- Test GPU device access, model-directory permissions, downloads, and daemon/client communication.
- Keep model weights outside the application bundle.
- Avoid broad filesystem access where a dedicated application/model directory suffices.
- Document GPU-specific caveats and CPU fallback.
- Test both fresh installation and upgrade behavior.
- Do not assume that setting an environment variable can enable a backend that was not compiled or packaged.

## 12. Testing and acceptance criteria

Each phase must include tests appropriate to its scope. Tests must be reproducible and report failures honestly.

### General quality

- No unexplained compiler warnings in project-owned code.
- No hard-coded assumptions about a CPU generation as a substitute for feature detection.
- Missing optional GPU libraries must not break the CPU-only build.
- No silent fallback: log the selected backend and why fallback occurred.
- No claims of speed or memory improvement without recorded measurements.
- Test invalid inputs, allocation failures where practical, cancellation, and clean shutdown.
- Document upstream dependency versions and licenses.

### Representative test matrix

Test as available:
- CPU-only x86-64 machine.
- AVX2-capable CPU.
- Intel integrated GPU.
- Intel Arc or another capable discrete GPU.
- At least one non-Intel Vulkan device if cross-vendor Vulkan support is a goal.
- Low-memory configuration with an 8 GB limit.
- Flatpak sandboxed runtime.

Not all hardware needs to be available to complete early phases. Clearly mark untested configurations rather than simulating successful results.

### 8 GB constrained test

Use a controlled test environment, such as cgroups where supported, to limit memory. Record the exact configuration and distinguish:
- Process RSS.
- Cgroup memory usage.
- Swap usage.
- GPU/shared-memory behavior.
- OOM termination and logs.

A test passing under a memory cap does not by itself prove that all 8 GB machines can run the same model.

## 13. Phased implementation and approval gates

**Gemini must implement only the current phase and stop for approval before proceeding.** Each phase response must include files changed, complete code for new/modified files (or a clear repository patch), build/test commands, actual test results, known limitations, and a proposed next step.

### Phase 0 — Repository foundation and capability probe

Deliver:
- Repository layout and CMake configuration.
- Portable baseline build.
- Optional build presets for CPU optimizations and GPU backends, with unsupported options clearly handled.
- Shared C capability-report API and command-line probe.
- Unit tests for parsing/formatting and graceful behavior when optional information is absent.
- README with build instructions and documented assumptions.

Acceptance:
- Clean configure/build on a supported Linux environment.
- CPU-only build works without Vulkan/OpenCL/SYCL installed.
- Probe runs without elevated privileges and reports unavailable fields safely.
- CPU capabilities are not inferred solely from generation labels.
- Build presets do not silently produce binaries requiring unsupported CPU instructions.
- Gemini stops and requests approval.

### Phase 1 — `llama.cpp` integration and baseline benchmark

Deliver:
- One pinned `llama.cpp` dependency/integration.
- Small CLI to load a known GGUF model, run a prompt, stream output, and exit cleanly.
- CPU-only baseline.
- Optional Vulkan build and runtime check.
- Documented benchmark script/results format.
- OpenCL/SYCL evaluation plan, with implementation only where dependencies and target hardware justify it.

Acceptance:
- Model load, inference, error path, and clean shutdown work.
- Output and token counts are recorded.
- Prefill and decode metrics are separate.
- Backend selection and fallback are logged.
- Gemini stops and requests approval.

### Phase 2 — Memory budget and lifecycle

Deliver:
- Explicit load/unload lifecycle.
- Context and memory configuration.
- Memory estimates and measurements.
- Stress tests for repeated load/unload and allocation failures where practical.
- A report comparing observed behavior with the baseline.

Acceptance:
- No known resource leaks across repeated lifecycle tests.
- Memory claims include model, quantization, context, backend, OS, and measurement method.
- No custom streaming/pinning added without benchmark evidence.
- Gemini stops and requests approval.

### Phase 3 — Daemon and local API

Deliver:
- `ardad` process.
- Versioned local API with streaming generation and cancellation.
- Capability and telemetry endpoints.
- Error handling and API documentation.
- Integration tests for disconnect, cancellation, and shutdown.

Acceptance:
- Local-only binding by default.
- UI clients cannot block inference or daemon shutdown indefinitely.
- API input and model paths are validated.
- Gemini stops and requests approval.

### Phase 4 — Model registry and download manager

Deliver:
- Repository search and metadata display.
- Resumable downloads, cancellation, integrity checks where available, and temporary-file handling.
- Hardware-aware memory estimates with uncertainty clearly labeled.
- Tests for interrupted and invalid downloads.

Acceptance:
- No downloaded code is executed.
- Incomplete downloads are not shown as valid installed models.
- License and provenance metadata are displayed when available.
- Gemini stops and requests approval.

### Phase 5 — GTK4 / Libadwaita client

Deliver:
- Chat UI, model selection, streaming, cancel, download UI, and system status.
- Non-blocking communication with the daemon.
- UI error and empty states.
- Measured memory footprint.

Acceptance:
- Inference and network operations do not block the GTK main loop.
- Client disconnect/reconnect and daemon-unavailable states are handled.
- Gemini stops and requests approval.

### Phase 6 — Flatpak and final validation

Deliver:
- Flatpak manifest and reproducible packaging instructions.
- Sandboxed GPU/backend checks.
- Installation, upgrade, model storage, and CPU fallback tests.
- Final README and hardware compatibility matrix.

Acceptance:
- Native and Flatpak builds both pass their applicable tests.
- Optional backend failures do not break CPU inference.
- Known limitations are documented.
- Gemini stops and requests final approval.

## 14. Instructions to Gemini

You are the implementation agent for Arda's Breath. Act as a senior systems engineer, C developer, CMake maintainer, and inference-runtime integrator.

Follow these rules:

1. **Implement Phase 0 only initially. Do not begin Phase 1 until the user explicitly approves Phase 0.**
2. Before writing code, inspect the current repository and available toolchain. If the repository is empty, create the proposed structure. Do not overwrite existing work without explaining the change and obtaining approval.
3. Check current upstream `llama.cpp` build options and documentation when beginning its integration. Do not invent CMake options, environment variables, backend controls, or API functions.
4. Prefer small, testable modules and standard C interfaces. Do not introduce dependencies without explaining why they are necessary.
5. Make build configurations explicit. Optional backend absence must not prevent a portable CPU build.
6. Do not claim code compiled, tests passed, a backend works, or performance improved unless that was actually verified in the available environment.
7. If hardware is unavailable, provide the test procedure and mark the result as untested.
8. Explain non-obvious architectural decisions and meaningful trade-offs, but avoid speculative optimization.
9. Keep code, build instructions, and test commands complete enough for the user to reproduce.
10. Do not silently broaden scope or skip approval gates.
11. When a requirement is ambiguous, choose the smallest reversible implementation and document the assumption. Ask the user only when the decision materially affects architecture or safety.
12. End each phase with a concise review summary, exact commands to reproduce tests, limitations, and a request for approval to proceed.

## 15. First task

Begin with **Phase 0 only**.

1. Inspect the repository and host build environment.
2. Propose the exact Phase 0 file list and key design decisions.
3. Implement the capability probe and portable CMake foundation.
4. Build and run tests where the environment permits.
5. Present the resulting files, commands, verified results, and known limitations.
6. Stop and wait for explicit user approval before starting Phase 1.

Do not implement the daemon, model manager, GTK UI, model inference, GPU kernels, custom disk streaming, or Flatpak packaging in Phase 0.
