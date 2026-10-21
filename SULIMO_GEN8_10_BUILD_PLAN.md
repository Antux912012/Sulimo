# Súlimo (Arda's Breath): Build Plan for Gemini 3.8 Flash
## Gen8–10 Legacy Tier, Rust engine v0.1.0

> **Audience:** Gemini 3.8 Flash, the implementing coding agent.
> **Owner/Reviewer:** Antonio (human approval required at every milestone gate).
> **Source spec:** "MASTER BUILD SPECIFICATION: Sulimo (Arda Breath), Gen8–10 Legacy Tier (v0.1.0)".
> This plan **refines** that spec. Where they differ, follow this plan: §2 explains every change.

---

## 0. How to use this document (agent instructions)

1. Read §1–§4 completely before writing any code. Load §5 (DSL legend) into context for every session.
2. Implement **exactly one milestone at a time** (§7). Do not start the next milestone until the human replies `APPROVED Mx`.
3. At the end of every milestone, output the **Milestone Report** (§10) using **real** command output. Never invent benchmark numbers, test results, or hardware behavior. If something could not be tested (missing hardware/driver), write `UNTESTED:` and explain why.
4. Hard engineering rules (apply to all milestones):
   - `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` must pass.
   - Every `unsafe` block has a `// SAFETY:` comment stating the invariant it relies on.
   - Every SIMD or OpenCL kernel has a **scalar Rust reference implementation** and a test comparing both.
   - No new crate dependency without a one-line justification in the Milestone Report.
   - Missing OpenCL drivers or GPU must **never** crash or fail the build. The CPU path must always work.
   - Never shell out with user-supplied strings. Never execute anything downloaded with a model.
   - No allocation on the per-token hot path after model load (the static arena owns all buffers).

---

## 1. Mission & scope

Súlimo is a local, offline LLM inference engine with a native GTK4/Libadwaita client. It targets **Intel 5th–10th gen laptops** (Broadwell → Comet Lake / Ice Lake) with **8 GB RAM**, using:

- **iGPU (OpenCL, FP16, SIMD8)** for **prompt prefill** (batch of N ≥ threshold tokens, compute-bound).
- **CPU physical cores (AVX2 + FMA)** for **token decode** (N = 1, memory-bandwidth-bound), while the iGPU idles into **RC6**.
- A hard **5.2 GB** unified-memory ceiling for the engine (≈2.8 GB left for OS + UI on an 8 GB host).

**In scope v0.1.0:** GGUF v3 loading, Llama-family decoder-only models (Llama 3.x; Qwen2.x as second arch), Q4_0 weights, Q8_0 KV cache, greedy/temperature/top-k/top-p sampling, GTK4 chat UI, telemetry, `.deb` package.

**Out of scope v0.1.0:** K-quants (Q4_K etc.), MoE, multimodal, Jinja template engine, network API/daemon, Vulkan/SYCL, AVX-512 paths, cloud anything, Windows/macOS.

---

## 2. Spec amendments (read carefully)

Each item explains what changed from the original spec and why. Implement the **amended** version.

| # | Original spec | Amendment | Reason |
|---|---|---|---|
| A1 | 4 crates | Add `sulimo-engine` (forward pass, scheduler, sampler) and `sulimo-cli` (headless runner/bench). | `core`/`ocl`/`cpu` must not depend on each other. Something must orchestrate them, and you need a headless binary for tests and benchmarks. |
| A2 | Spec omits tokenizer & forward pass | Add explicit milestone for **tokenizer + full transformer forward pass** (RMSNorm, RoPE, GQA attention, SwiGLU, sampling, chat template). | Without these, Q4_0 matmul kernels alone do not produce text. |
| A3 | Milestone order MMAP → OCL → CPU → UI | **MMAP → CPU reference engine → OCL → Hybrid scheduler → UI → Telemetry/bench → Packaging.** | The CPU path is the **correctness oracle** for the GPU kernels and the fallback on machines without OpenCL. Writing GPU kernels first leaves you with nothing to validate them against. |
| A4 | "Enforce 4096-byte alignment on tensor offsets" in the file | GGUF tensor offsets are only aligned to `general.alignment` (default **32 B**). You can't enforce 4096 on arbitrary files. **Load design:** stream tensors from the mmap into a **page-aligned anonymous arena**, placing each tensor at a 4096-B boundary with size padded to a 64-B multiple, then wrap with `CL_MEM_USE_HOST_PTR`. | Intel zero-copy requires host ptr 4096-aligned + size multiple of 64 B. This satisfies `$ZCOPY` for *any* GGUF file. |
| A5 | `MADV_HUGEPAGE` on the GGUF mmap | Apply `MADV_HUGEPAGE` to the **anonymous arena** (2 MiB-aligned). Use `MADV_SEQUENTIAL`/`MADV_WILLNEED` on the file mmap and `posix_fadvise(DONTNEED)` after copying each chunk. | THP on regular-file mappings is a no-op on most filesystems (ext4). Dropping page cache keeps peak memory under budget during load. |
| A6 | Q4_0 raw 18-byte blocks | At load, **repack Q4_0 tensors to SoA**: contiguous `f16 d[]` array + contiguous `u8 qs[]` array (each 64-B aligned). Both CPU and GPU kernels consume SoA. | 18-byte AoS blocks break aligned vector loads on CPU and sub-group loads on GPU. Repacking is free because A4 already copies the data. |
| A7 | "matrix-vector kernel" for prefill | Prefill with N ≥ 8 tokens is **GEMM** (`[N×K]·[K×M]`). Implement a tiled Q4_0×F16 GEMM. A matvec variant is optional, for comparison only. | Running matvec N times re-reads the weights N times, which wastes memory bandwidth. |
| A8 | FP16 everywhere on GPU | Multiply in `half`/`half4`, **accumulate in `float`** per output. `-cl-fast-relaxed-math` only for matmul/elementwise programs, **not** for softmax/attention. | FP16 accumulation over K=2048–14336 loses too much precision. Relaxed math on `exp` can break softmax. |
| A9 | Futex vs eventfd (spec mentions both) | Engine thread waits with `std::thread::park()`. The OpenCL `clSetEventCallback(CL_COMPLETE)` callback sets an `AtomicBool` and calls `Thread::unpark()` (futex-backed on Linux). Never call `clFinish`/`clWaitForEvents` on the hot path. | Same zero-spin effect with no `unsafe` futex syscalls. The callback must be quick and must not call blocking CL APIs. |
| A10 | "num_cpus::get_physical()" for pinning | `num_cpus` gives only a **count**. Choose **one logical CPU per physical core** from `/sys/devices/system/cpu/cpu*/topology/{core_id,thread_siblings_list}` and pin with `sched_setaffinity` (`libc`). | Spec intent (no SMT siblings) needs actual CPU IDs. |
| A11 | KV cache "180–280 MB for 4K–8K" fixed | **Compute KV size from model metadata:** `bytes/token = 2 · n_layer · n_head_kv · head_dim · 34/32`. Clamp context so weights + KV + scratch ≤ budget. | Example: Llama-3.2-3B ≈ 61 KB/token (4K ≈ 245 MB, 8K ≈ 490 MB). Qwen2.5-7B ≈ 30 KB/token. A fixed number is wrong for most models. |
| A12 | "Circular sliding-window KV" | Allocate as a ring buffer. **v0.1 eviction policy = context re-pack:** keep system prompt + most recent turns, then **re-prefill on the iGPU**. In-place StreamingLLM sliding (attention sinks + RoPE re-rotation of quantized K) is a **stretch goal** (M8), gated by quality tests. | Re-rotating Q8_0 keys adds error. Naive eviction without sinks degrades output. Re-prefill is simple, correct, and uses the fast GPU path. |
| A13 | "iGPU RC6 deep sleep" | RC6 cannot be commanded. Requirement = **no GPU submissions and no GPU polling during decode**. Verify via i915 `rc6_residency_ms` in sysfs. | RC6 entry is automatic when the GT is idle. |
| A14 | "25+ tok/s streaming" | This is a **UI capability** requirement (UI must render ≥25 tok/s at 60 FPS). Engine decode speed is bounded by memory bandwidth (see §3.3), so do not promise 25 tok/s for 3B+ models. | Honest performance claims. |
| A15 | "Estimated package power" | Read RAPL `energy_uj` **if readable**. Since kernel 5.10 it is usually root-only. Fallback: `/sys/class/power_supply/BAT*/power_now` (system draw, battery only). Else show "n/a" with an explanatory tooltip. Never require root. | Linux permission reality. |
| A16 | `intel-compute-runtime-legacy1` assumed | Detect at runtime: platform/device name, `cl_khr_fp16`, `cl_intel_subgroups`, `cl_intel_required_subgroup_size`. Provide a **portable kernel variant** without Intel attributes. If FP16 is missing, use CPU-only. **Broadwell (Gen8) driver support must be verified.** It may need an older NEO release or fall back to CPU-only. | Driver support for Gen8–Gen11 varies by release/distro. Mesa rusticl may also appear, without Intel extensions. |
| A17 | Celeron/Pentium of these gens | Many lack AVX2/FMA. Detect with `is_x86_feature_detected!`. Use a scalar CPU fallback (slow, but correct) and warn in UI. | Target-range accuracy. |
| A18 | "No heap allocation" in parser | Parser borrows all strings/arrays **zero-copy from the mmap**. Small bookkeeping `Vec`s (tensor index, vocab slice table) are allowed. **Tensor data is never copied except once into the arena (A4).** | A vocab of 128k entries needs an index. "No heap" is meant for tensor data. |
| A19 | Version v0.1.0 | The PPA already publishes `sulimo 0.1.0` (C `arda-probe`). The Rust engine `.deb` must be **≥ 0.2.0**, or the human must explicitly approve reusing 0.1.0. | Avoid APT version collision. |

---

## 3. Hardware targets, drivers, and realistic expectations

### 3.1 Target matrix

| Gen | CPU family | iGPU (arch) | Example EUs | RAM typical | Notes |
|---|---|---|---|---|---|
| 5th | Broadwell | HD 5500/6000, Iris 6100 (Gen8) | 23–48 | DDR3L-1600 | Verify OpenCL driver support (A16) |
| 6th | Skylake | HD 520/530 (Gen9) | 24 | DDR4-2133 | |
| 7th | Kaby Lake | HD 620/630 (Gen9.5) | 24 | DDR4-2133 | **Dev machine: i5-7200U + HD 620, 15.4 GB RAM** |
| 8th–9th | Kaby Lake-R / Coffee / Whiskey | UHD 620/630 (Gen9.5) | 24 | DDR4-2400 | |
| 10th | Comet Lake | UHD 620/630 (Gen9.5) | 24 | DDR4-2666 | |
| 10th | Ice Lake | Iris Plus G4/G7 (Gen11) | 48–64 | LPDDR4X-3733 | CPU also has AVX-512 (out of scope v0.1) |

### 3.2 Dev-machine prerequisites (human runs these; agent documents them)

The Phase-0 C probe on the dev machine reported **"No OpenCL platforms or ICDs installed"**, so OpenCL work (M3+) is blocked until this is fixed:

```bash
sudo apt install ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
sudo apt install intel-opencl-icd     # if HD 620 is no longer listed, install the legacy1 NEO build instead
clinfo | grep -E "Platform Name|Device Name|cl_khr_fp16|cl_intel_subgroups|cl_intel_required_subgroup_size"
```

Also needed: Rust stable (rustup), `pkg-config`, `libgtk-4-dev` (≥ 4.16), `libadwaita-1-dev` (≥ 1.6), and llama.cpp (built locally, for golden reference data only; it is not linked).

### 3.3 Realistic performance envelope (upper bounds, NOT targets)

Decode is memory-bandwidth-bound: `max tok/s ≈ effective_BW / bytes_read_per_token`. Q4_0 = 18 B per 32 weights (0.5625 B/weight). Assume dual-channel DDR4-2133 (≈34 GB/s peak, ~70% effective):

| Model (pure Q4_0) | Weights | Decode ceiling (dev machine) |
|---|---|---|
| Llama-3.2-1B | ≈0.75 GB | ≈30 tok/s |
| Llama-3.2-3B | ≈1.9 GB | ≈12 tok/s |
| Llama-3.1-8B | ≈4.7 GB | ≈5 tok/s (barely fits 5.2 GB with small context) |

Measured numbers go in the Milestone Report. Always compare against `llama-bench` on the same machine/model as a sanity reference.

---

## 4. Architecture

### 4.1 Workspace layout

The Rust workspace **coexists** with the existing C/CMake Phase-0 code at the repo root until M7. See §4.4 for why this is mandatory.

```text
Súlimo/
├── Cargo.toml                    # [workspace] resolver = "2"; [workspace.package] version = "0.2.0"
├── rust-toolchain.toml           # channel = "stable"
├── crates/
│   ├── sulimo-core/              # @C:MMAP + @C:TOK: GGUF v3, arena, budget, KV cache, tokenizer, model config
│   ├── sulimo-cpu/               # @C:CPU: topology, pinning, thread pool, AVX2/FMA + scalar kernels
│   ├── sulimo-ocl/               # @C:OCL-GEN9: device probe, program build/cache, kernels, zero-spin waiter
│   ├── sulimo-engine/            # @C:ENG: forward pass, hybrid scheduler, sampler, chat templates, telemetry
│   ├── sulimo-cli/               # bin `sulimo-cli`: probe | run | bench | golden-compare
│   └── sulimo-gtk/               # @UI:GTK: bin `sulimo`: GTK4 + Libadwaita client
├── kernels/                      # *.cl sources (embedded with include_str!)
├── tests/golden/                 # small golden fixtures (large ones gitignored, regenerated by script)
├── scripts/                      # gen_golden.sh, bench.sh, rc6_check.sh
├── CMakeLists.txt, shared/, tools/, tests/unit/ …   # existing C Phase-0 code: DO NOT DELETE before M7
└── packaging/
```

### 4.2 Dependency rules (enforced)

```text
sulimo-gtk ─┐
sulimo-cli ─┴─> sulimo-engine ─┬─> sulimo-core
                               ├─> sulimo-cpu ──> sulimo-core (types only)
                               └─> sulimo-ocl ──> sulimo-core (types only)
```

- `sulimo-cpu` and `sulimo-ocl` never depend on each other.
- `sulimo-gtk` never touches kernels/arena directly. It talks only to the `sulimo-engine` public API over channels.
- `sulimo-ocl` is behind a cargo feature `opencl` (default **on**). It must also degrade gracefully at **runtime** when `libOpenCL.so.1` or an ICD is missing. Prefer runtime loading (e.g. a `dynamic` feature of `cl3`/`opencl3` if available, otherwise `libloading`).

### 4.3 Indicative dependencies (pin exact versions at M0; record in Milestone Report)

| Crate | Purpose |
|---|---|
| `memmap2` | file mmap + `advise()` |
| `libc` | `mmap` anon arena, `madvise`, `posix_fadvise`, `sched_setaffinity` |
| `half` | f16 ↔ f32 on CPU |
| `opencl3` (or `cl3`) | OpenCL 3.0 bindings |
| `flume` | engine ↔ UI channels |
| `gtk4` (feature `v4_16`), `libadwaita` (feature `v1_6`) | UI |
| `fancy-regex` | BPE pre-tokenizer regexes (lookahead) |
| `thiserror` (libs), `anyhow` (bins only) | errors |
| `rand` + `rand_chacha` | seeded sampling |
| `criterion` (dev) | micro-benchmarks |
| `num_cpus` | sanity count only (A10) |

### 4.4 Repository & pipeline continuity (critical)

- The Arda-Tech PPA auto-updater (`~/arda-tech-ppa/scripts/auto-update-ppa.sh`, systemd timer every ~10 min) builds Súlimo by running `./packaging/build_deb.sh` and reads the version with `grep "project(arda_breath" CMakeLists.txt`. The script runs under `set -euo pipefail`.
- **If root `CMakeLists.txt` disappears or loses that line, the version grep fails and the whole PPA updater aborts for *all* apps.**
- Therefore, until M7: keep `CMakeLists.txt`, `packaging/build_deb.sh` and the C probe building exactly as today. Add `target/` to `.gitignore`. Develop on branch `rust-engine` (local repo has no remote yet).
- M7 updates `build_deb.sh` and the updater's `get_source_version` for `sulimo` to read `[workspace.package] version` from `Cargo.toml`, in the **same change**.

### 4.5 Runtime data flow

```text
UI (GTK main loop) ──flume Cmd──> Engine thread ──┬─ Prefill (N ≥ T): enqueue all layer kernels on iGPU, park() until CL callback
          ^                                       │       └─ writes Q8_0 K/V into shared ring KV (USE_HOST_PTR)
          └──flume Event (Token/Telemetry/Done)── └─ Decode (N = 1): CPU pool on physical cores, AVX2+FMA; GPU idle → RC6
```

Threshold `T` defaults to 8 and is configurable. It is calibrated in M6.

---

## 5. #SULIMO-DSL v1.1: legend (paste into every agent session)

**Modules `@`**

| Code | Meaning |
|---|---|
| `@C:MMAP` | GGUF v3 parser, zero-copy mmap, static arena, budget (sulimo-core) |
| `@C:TOK` | tokenizer + chat templates (sulimo-core) |
| `@C:CPU` | topology/pinning, thread pool, AVX2+FMA + scalar kernels (sulimo-cpu) |
| `@C:COL` | Q8_0 ring KV cache (sulimo-core) |
| `@C:ENG` | forward pass, scheduler, sampler (sulimo-engine) |
| `@C:OCL-GEN9` | OpenCL Gen8–11 FP16 SIMD8 kernels + zero-spin waiter (sulimo-ocl) |
| `@UI:GTK` | GTK4/Libadwaita client (sulimo-gtk) |

**Actions `!`**

| Code | Meaning |
|---|---|
| `!DIFF` | Output only unified diffs against current files |
| `!KERN` | Output raw OpenCL C kernel + its Rust binding + scalar reference + test |
| `!FULL` | Output the complete functional module (all files) |
| `!TEST` | Output tests + exact commands to run them, then actual results |
| `!RPT` | Output the Milestone Report (§10) only |

**Constraints `$`**

| Code | Meaning |
|---|---|
| `$ZCOPY` | Arena tensors 4096-B aligned, size padded to 64 B, `CL_MEM_USE_HOST_PTR`, verify no copy |
| `$BATT` | Decode on one logical CPU per physical core, pinned. No GPU submits/polling during decode. Zero-spin GPU waits |
| `$MEM<8G` | All engine memory ≤ `MAX_RAM_BUDGET_BYTES = 5_200 * 1024 * 1024`. Refuse to load otherwise |
| `$NOALLOC` | No heap allocation on the per-token hot path |
| `$REF` | Must match scalar reference / llama.cpp golden within stated tolerance |

**Example prompts**

```text
@C:MMAP !FULL $ZCOPY $MEM<8G — implement M1 per SULIMO_GEN8_10_BUILD_PLAN.md §7. Stop at gate.
@C:OCL-GEN9 !KERN $ZCOPY $REF — M3a Q4_0×F16 GEMM. Include portable variant (A16).
@C:CPU !DIFF $BATT $NOALLOC — fix thread pool to park instead of spin after 2 µs.
```

---

## 6. Technical reference (use exactly; do not guess)

### 6.1 GGUF v3 (little-endian only; reject big-endian/other versions)

```text
u32 magic = 0x46554747 ("GGUF")   u32 version = 3
u64 tensor_count                  u64 metadata_kv_count
metadata_kv[]: string key; u32 value_type; value
  types: 0 U8,1 I8,2 U16,3 I16,4 U32,5 I32,6 F32,7 BOOL,8 STRING,9 ARRAY(u32 elem_type,u64 len,elems),10 U64,11 I64,12 F64
  string = u64 len + UTF-8 bytes (no NUL)
tensor_info[]: string name; u32 n_dims; u64 dims[n_dims]; u32 ggml_type; u64 offset (relative to data start)
data_start = align_up(end_of_tensor_infos, general.alignment (default 32))
ggml_type: 0 F32, 1 F16, 2 Q4_0, 8 Q8_0, 14 Q6_K, 30 BF16 (v0.1 supports F32, F16, Q4_0, Q8_0; reject others with a clear message)
```

Validate every offset/size against the file length (no out-of-bounds slices from malformed files). Fuzz the parser (`cargo fuzz` optional, or proptest on truncated/corrupted headers).

### 6.2 Quant blocks

```text
Q4_0 block (18 B, 32 weights): f16 d; u8 qs[16]
   w[i]    = ((qs[i] & 0x0F) - 8) * d   for i in 0..16
   w[i+16] = ((qs[i] >> 4)   - 8) * d   for i in 0..16
Q8_0 block (34 B, 32 values):  f16 d; i8 qs[32];  v[i] = qs[i] * d;  d = max|x| / 127
```

CPU dot product Q4_0·Q8_0 (decode): quantize activations to Q8_0 once per matvec. Per block, integer dot via `_mm256_maddubs_epi16` / `_mm256_madd_epi16` (sign trick as in llama.cpp `ggml_vec_dot_q4_0_q8_0`). Convert to f32 and accumulate `d_w * d_x * sumi` with **`_mm256_fmadd_ps`**. Gate with `#[target_feature(enable = "avx2,fma")]` + runtime `is_x86_feature_detected!` dispatch.

### 6.3 Llama-family forward pass (per token position)

```text
x = token_embd[tok]
for each layer l:
  h = rmsnorm(x, eps) * attn_norm
  q = Wq·h, k = Wk·h, v = Wv·h            (qwen2: + bq, bk, bv biases)
  rope(q, k, pos)                          llama: NORM mode (adjacent pairs); qwen2: NEOX mode (split halves)
                                           llama3: apply rope_freqs.weight factors if tensor present
  store k, v → KV[l][pos] as Q8_0
  a = softmax(q·Kᵀ / sqrt(head_dim) + causal_mask) · V      (GQA: head h uses kv head h / (n_head/n_head_kv))
  x = x + Wo·a
  h = rmsnorm(x, eps) * ffn_norm
  x = x + Wdown·(silu(Wgate·h) ⊙ (Wup·h))
logits = Wout·(rmsnorm(x, eps) * output_norm)          (Wout = token_embd if output.weight absent / tied)
```

GGUF names: `token_embd.weight`, `blk.{l}.attn_norm.weight`, `blk.{l}.attn_{q,k,v,output}.weight`, `blk.{l}.ffn_norm.weight`, `blk.{l}.ffn_{gate,up,down}.weight`, `output_norm.weight`, `output.weight` (optional), `rope_freqs.weight` (optional).
Metadata: `{arch}.context_length, .embedding_length, .block_count, .feed_forward_length, .attention.head_count, .attention.head_count_kv, .attention.layer_norm_rms_epsilon, .rope.freq_base, .rope.dimension_count`; `tokenizer.ggml.{model,pre,tokens,scores,token_type,merges,bos_token_id,eos_token_id}`, `tokenizer.chat_template`.

### 6.4 Reference models (development)

Produce **pure** Q4_0 files from F16 GGUF with llama.cpp: `llama-quantize --pure model-f16.gguf model-q4_0-pure.gguf Q4_0`.
1. `Llama-3.2-1B-Instruct` (fast iteration), 2. `Llama-3.2-3B-Instruct` (primary target), 3. `Qwen2.5-1.5B/3B-Instruct` (second arch, M2 stretch).

---

## 7. Milestones

Each milestone ends with a **GATE**: produce the Milestone Report and stop.

### M0: Workspace bootstrap & Rust capability probe
**Tags:** `@C:CPU @C:OCL-GEN9 !FULL`

Deliver:
- Branch `rust-engine`. Root `Cargo.toml` workspace, `rust-toolchain.toml`, crate skeletons per §4.1, `.gitignore` += `target/`.
- `sulimo-cpu::topology`: CPU vendor/model, feature flags (SSE4.2, AVX, AVX2, FMA, F16C, AVX-512F), logical→physical core map (A10), L3 size from sysfs.
- `sulimo-ocl::probe`: runtime OpenCL discovery: platforms, devices, driver version, EU count (`CL_DEVICE_MAX_COMPUTE_UNITS`), extensions (A16), `CL_DEVICE_MEM_BASE_ADDR_ALIGN`, host-unified-memory flag. Missing loader/ICD → `Unavailable(reason)`, never panic.
- `sulimo-core::sysmem`: MemTotal/MemAvailable/Swap from `/proc/meminfo`, THP mode from `/sys/kernel/mm/transparent_hugepage/enabled`.
- `sulimo-cli probe [--json]` printing a capability report and the selected tier: `hybrid-gen9` | `cpu-avx2` | `cpu-scalar`, with a human-readable reason.
- Basic CI-style script `scripts/check.sh` (fmt, clippy, test).

Acceptance:
- Builds and runs with **and** without an OpenCL ICD installed (test by unsetting/renaming `OCL_ICD_VENDORS` dir or `--no-default-features`).
- On the dev machine reports: 2 physical / 4 logical, AVX2+FMA yes, AVX-512 no, and HD 620 details if the ICD is installed.
- C Phase-0 build + 5 CTest tests still pass (`ctest --test-dir build/portable`), and `packaging/build_deb.sh` still works.
- **GATE M0.**

### M1: GGUF v3 parser, static arena & budget
**Tags:** `@C:MMAP !FULL $ZCOPY $MEM<8G`

Deliver:
- `gguf::GgufFile<'a>`: mmap (`memmap2`, read-only, `MADV_SEQUENTIAL`+`WILLNEED`), header/KV/tensor-info parsing per §6.1, zero-copy borrowed metadata (A18), typed getters (`get_u32`, `get_str`, `get_arr_str`…), tensor index by name.
- `ModelConfig::from_gguf()` for `llama` and `qwen2` archs (§6.3 keys), validation of shapes vs config.
- `arena::Arena`: one anonymous `mmap` (2 MiB-aligned, `MADV_HUGEPAGE`), bump allocator with **4096-B** alignment and **64-B** size padding per allocation (A4). Regions: weights, KV, scratch/activations. Unmapped on drop.
- `budget::MemoryBudget`: `MAX_RAM_BUDGET_BYTES = 5_200 * 1024 * 1024`. **Pre-flight plan**: weights (post-repack) + KV (A11) + scratch + 5% slack. If over budget, try smaller context. If still over, return `BudgetError` with an itemized breakdown.
- Loader: stream each tensor from the mmap into the arena; Q4_0 → SoA repack (A6); F32/F16 copied as-is; `posix_fadvise(DONTNEED)` per chunk (A5).
- `sulimo-cli inspect model.gguf` (metadata, tensor table, budget plan) and `sulimo-cli load model.gguf` (load + report RSS).

Acceptance:
- Parser unit tests: hand-built tiny GGUF fixtures (valid, truncated, bad magic, bad version, OOB offset, unsupported type) → correct Ok/Err, no panics.
- Repack round-trip test: dequantize(AoS) == dequantize(SoA) bit-exact.
- Loading Llama-3.2-3B pure Q4_0: every arena tensor pointer `% 4096 == 0`, every padded size `% 64 == 0` (asserted in test).
- Peak RSS during load ≤ planned budget + 64 MB (measure via `/proc/self/status` VmHWM). Run also under `systemd-run --user --scope -p MemoryMax=5.2G sulimo-cli load …` and report the result.
- **GATE M1.**

### M2: CPU reference engine: tokenizer, forward pass, KV, AVX2 decode
**Tags:** `@C:TOK @C:CPU @C:COL @C:ENG !FULL $BATT $NOALLOC $REF`

Deliver:
- **Tokenizer** (`@C:TOK`): byte-level BPE from GGUF metadata (`tokenizer.ggml.model = "gpt2"`) with pre-tokenizer regex selected by `tokenizer.ggml.pre` (`llama-bpe`, `qwen2`). Encode/decode, special tokens, BOS/EOS. SentencePiece (`"llama"` model) is optional stretch.
- **Chat templates:** hard-coded `llama3` and `chatml` formats selected from arch/pre. **No Jinja.**
- **Scalar reference kernels** (f32): rmsnorm, rope (NORM/NEOX + rope_freqs), softmax, silu, matvec over dequantized Q4_0/Q8_0/F16/F32, attention.
- **AVX2+FMA kernels:** Q4_0(SoA)·Q8_0 matvec (§6.2), F16/F32 matvec, rmsnorm, softmax, attention over Q8_0 KV. Runtime dispatch with scalar fallback (A17).
- **Thread pool** (`$BATT`): one worker per physical core, pinned (A10). Rows split across workers. Barrier = bounded spin (≤ ~2 µs, `std::hint::spin_loop`) then `park()`. Configurable.
- **KV cache** (`@C:COL`): Q8_0 ring buffer in the arena, sized by A11, re-pack policy (A12) on overflow.
- **Engine API (`sulimo-engine`):** `Engine::load(path, opts)`, `generate(prompt, SamplingParams, cancel: &AtomicBool) -> impl Iterator<Item=Token>`. Sampler: greedy, temperature, top-k, top-p, repetition penalty, seeded ChaCha RNG.
- `sulimo-cli run model.gguf -p "…" [--greedy] [--threads N] [--ctx N]` streaming to stdout, printing prefill/decode tok/s, TTFT, VmHWM at exit.
- `scripts/gen_golden.sh`: uses local llama.cpp to generate (a) token IDs for 20 test strings via `llama-tokenize`, (b) greedy continuations (32 tokens) for 5 prompts, (c) logits for the first decode step if feasible. Run llama.cpp with `-ctk q8_0 -ctv q8_0` for KV parity (requires flash-attn in llama.cpp; if unavailable, document and compare with default KV).

Acceptance (`$REF`):
- Tokenizer: 100% exact match with `llama-tokenize` on all fixtures (incl. emoji, CJK, code, whitespace runs).
- AVX2 kernels vs scalar: max relative error ≤ 1e-3 on random inputs (Q8_0 activation quantization is part of both).
- Greedy decode: identical first 32 tokens to llama.cpp on ≥ 4/5 golden prompts for Llama-3.2-1B and 3B pure Q4_0. Report any divergence position and logit gap.
- No heap allocation per token after warm-up (test with a counting `#[global_allocator]` in a test binary).
- Report measured decode tok/s vs §3.3 ceiling and vs `llama-bench` (same threads, same model).
- **GATE M2.**

### M3a: OpenCL Gen8–11 Q4_0×F16 GEMM + zero-copy + zero-spin waiter
**Tags:** `@C:OCL-GEN9 !KERN $ZCOPY $BATT $REF`
**Blocked until** `clinfo` shows the Intel GPU with `cl_khr_fp16` (§3.2).

Deliver:
- Program builder: embedded `kernels/gemm_q4_0.cl`. Build options `-DGEN8_PLUS_FP16 -cl-mad-enable -cl-fast-relaxed-math` (A8 scope). `#pragma OPENCL EXTENSION cl_khr_fp16 : enable`. Binary cache in `~/.cache/sulimo/ocl/<device-hash>.bin` via `CL_PROGRAM_BINARIES`, invalidated on driver version change.
- Kernel variants chosen at runtime (A16):
  1. `intel`: `__attribute__((intel_reqd_sub_group_size(8)))`, `half4` `fma`, sub-group shuffles/broadcast for activation reuse, FP32 accumulation.
  2. `portable`: same math, no Intel extensions.
- Buffers: one `cl_mem` per arena tensor with `CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY` (weights) / `READ_WRITE` (activations, KV). **Verify zero-copy:** `clEnqueueMapBuffer` returns the original host pointer (assert equality; log if the driver copies).
- Coherency: after GPU completion, the CPU reads only after a map/unmap pair (or documented equivalent) on shared buffers.
- Zero-spin waiter (A9): `clFlush` → `clSetEventCallback(CL_COMPLETE)` → `park()` loop on `AtomicBool`. Errors (`event status < 0`) propagate as `OclError`.

Acceptance:
- GEMM vs scalar reference for shapes N ∈ {8, 32, 128, 512} × (K, M) ∈ model shapes: max relative error ≤ 2e-3, cosine ≥ 0.9999.
- Zero-copy assertion passes on HD 620 (or reported with driver details if not).
- **Zero-spin proof:** while a long GEMM runs, the engine thread's CPU time (`/proc/self/task/<tid>/stat` utime+stime delta) ≤ 2% of wall time. Also report total process CPU time; if a driver thread spins, report it honestly.
- Report GFLOPS achieved vs theoretical (HD 620: 24 EU × 16 FP32 FLOP/clk × ~1.0 GHz ≈ 384 GFLOPS FP32, ~2× FP16).
- **GATE M3a.**

### M3b: Full-layer GPU prefill
**Tags:** `@C:OCL-GEN9 !KERN $ZCOPY $REF`

Deliver:
- Kernels: `rmsnorm`, `rope` (NORM/NEOX + freq factors), `kv_store_q8_0` (writes into the shared ring KV in the exact CPU layout), causal GQA `attention_prefill` (tiled, FP16 multiply / FP32 accumulate, numerically stable softmax **without** relaxed math), `silu_mul`, `add`, `gemm_q4_0` (from M3a), F16/F32 GEMM for non-Q4_0 tensors.
- Per ubatch: enqueue **all layers** on an in-order queue, **one** wait at the end (no host sync per layer). Final-position logits only.

Acceptance:
- GPU-prefill → CPU-decode produces the same greedy 32 tokens as CPU-only (M2) on ≥ 4/5 golden prompts. KV written by GPU dequantizes within 1 Q8_0 step of the CPU-written KV.
- Report prefill tok/s for 128/512/2048-token prompts: GPU vs CPU (M2).
- **GATE M3b.**

### M4: Hybrid asymmetric scheduler & battery discipline
**Tags:** `@C:ENG $BATT $MEM<8G`

Deliver:
- Scheduler: N ≥ T → GPU prefill (ubatch size configurable, default 512); N < T or no OpenCL → CPU. Decode always CPU. Context overflow → A12 re-pack + GPU re-prefill.
- Logging of every backend decision with its reason (no silent fallback).
- Cancellation: `AtomicBool` checked between ubatches and between tokens. Cancel latency ≤ one ubatch / one token.
- `scripts/rc6_check.sh`: samples i915 `rc6_residency_ms` (e.g. `/sys/class/drm/card*/gt/gt0/rc6_residency_ms` or `/sys/class/drm/card*/power/rc6_residency_ms`) during a 30 s decode.

Acceptance:
- During decode: GPU RC6 residency ≥ 90% of wall time (A13). Decode threads == physical cores, each pinned to a distinct physical core (verify via `/proc/<pid>/task/*/status` `Cpus_allowed_list`).
- During GPU prefill: CPU utilization of engine process ≤ 5% of one core (excluding driver threads, reported separately).
- 50× load/generate/unload cycle: no RSS growth > 16 MB, no leaked `cl_mem` (track counts).
- Under `systemd-run --user --scope -p MemoryMax=5.2G`: Llama-3.2-3B runs a 4K-context chat without OOM.
- **GATE M4.**

### M5: GTK4 + Libadwaita client
**Tags:** `@UI:GTK !FULL`

Deliver:
- `AdwApplication` (`app-id: io.github.antux912012.Sulimo`), `AdwApplicationWindow` → `AdwToolbarView` with `AdwHeaderBar` (model picker, new chat, menu) and bottom **Arda Telemetry Bar**.
- Chat view: message list (`GtkListView` or `GtkBox` in `GtkScrolledWindow`), input `GtkTextView` with Ctrl+Enter send, **Stop** button, auto-scroll that pauses when the user scrolls up.
- Engine runs on its own thread. `flume` channels: `Cmd {Load, Unload, Generate, Stop}` and `Event {Loaded, Token(String), Telemetry(..), Done(stats), Error(..)}`. UI consumes events in `glib::spawn_future_local` with `recv_async()`.
- **Frame-budget rendering:** coalesce tokens and flush to the text buffer at most once per frame (widget tick callback or ~16 ms timeout). Never insert per token.
- Telemetry Bar: `GtkLevelBar` for engine RAM vs 5.2 GB (offset marks at 80%/95%), power readout per A15 (label states source: "pkg (RAPL)" / "system (battery)" / "n/a"), phase indicator (Prefill-GPU / Decode-CPU / Idle), tok/s.
- Errors shown as `AdwToast` / `AdwBanner` (e.g. "OpenCL unavailable: running CPU-only", "Model exceeds 5.2 GB budget: itemized…").
- Desktop file, metainfo (AppStream, same style as other Arda-Tech apps), icons.

Acceptance:
- Synthetic stress: a fake engine emitting 100 tok/s keeps UI ≥ 58 FPS (measure with `GTK_DEBUG=interactive` FPS overlay or frame-clock timings logged). Report.
- UI never blocks during model load (load runs on engine thread, progress shown).
- UI process RSS (excluding engine arena) measured and reported.
- **GATE M5.**

### M6: Telemetry, benchmarks & threshold calibration
**Tags:** `@C:ENG !TEST`

Deliver:
- `sulimo-cli bench model.gguf --json`: prefill tok/s (128/512/2048), decode tok/s (128 tokens), TTFT, VmHWM, RC6 residency, energy/token if RAPL readable. Output schema documented in `benchmarks/README.md`.
- Calibrate T: sweep N ∈ {1,2,4,8,16,32} for GPU vs CPU prefill. Pick the crossover. Store per-device defaults.
- Comparison table vs `llama-bench` (same model/threads) on the dev machine.

Acceptance: JSON results committed under `benchmarks/results/<host>-<date>.json`. Claims in README cite these files only.
**GATE M6.**

### M7: Packaging & pipeline continuity
**Tags:** `!FULL`

Deliver:
- `packaging/build_deb.sh` rewritten for cargo (`cargo build --release --locked`), produces `sulimo_<ver>_amd64.deb` with `/usr/bin/sulimo`, `/usr/bin/sulimo-cli`, compatibility symlink `/usr/bin/arda-probe → sulimo-cli probe` wrapper, desktop/metainfo/icons. `Depends:` gtk4, libadwaita; `Recommends: intel-opencl-icd | mesa-opencl-icd, ocl-icd-libopencl1`.
- **Same commit:** update `~/arda-tech-ppa/scripts/auto-update-ppa.sh` `get_source_version` for `sulimo` to read `version` under `[workspace.package]` in `Cargo.toml`. Then (and only then) retire the C/CMake code (move to `legacy/` or delete with approval).
- Version per A19 (≥ 0.2.0 unless approved).
- Flatpak (exploratory, may ship CPU-only): `org.gnome.Platform//50` + `rust-stable` extension, `--device=dri`. **Risk:** the Intel NEO OpenCL ICD is not in the GNOME runtime. Evaluate: (a) Mesa rusticl from the GL extension (likely lacks Intel sub-group extensions → portable kernel), (b) bundling NEO, (c) CPU-only Flatpak + recommend `.deb` for iGPU. Report findings and do not block release on Flatpak.
- `apps.json` entry update in `arda-tech` / `arda-tech-ppa`.

Acceptance: `.deb` installs on Debian sid, `sulimo` launches, `sulimo-cli probe` works. A dry run of `auto-update-ppa.sh --no-push` completes without error. **GATE M7.**

### M8: Stretch (only after explicit approval)
StreamingLLM in-place sliding KV (sinks + RoPE re-rotation) with quality tests; SentencePiece tokenizer; Q4_K_M support; AVX-512 path for Ice Lake; Vulkan backend comparison.

---

## 8. Mapping: original spec → this plan

| Original | This plan |
|---|---|
| Milestone 1 (GGUF + arena) | M1 (+ A4/A5/A6/A18 fixes) |
| Milestone 2 (OpenCL FP16 SIMD8) | M3a + M3b (after the CPU oracle exists) |
| Milestone 3 (CPU AVX2 decoder + Q8_0 KV) | M2 (moved earlier; + tokenizer/forward pass) |
| Milestone 4 (GTK4 UI + telemetry bar) | M5 (+ M6 for telemetry/bench) |
| (missing) | M0 bootstrap, M4 scheduler, M7 packaging |

---

## 9. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Intel OpenCL ICD unavailable/unsupported for Gen8/Gen9 on Debian | No GPU prefill | CPU path complete in M2. Runtime detection. Document driver install. |
| Driver silently copies instead of zero-copy | 2× memory, budget breach | M3a assertion on mapped pointer. Budget includes worst case if copy detected. |
| Driver-internal polling thread burns CPU during waits | Battery goal missed | Measure per-thread CPU. Report honestly. Try `CL_QUEUE_PROFILING` off / different wait modes. |
| Tokenizer mismatch vs llama.cpp | Wrong outputs | Golden fixtures. Exact-match gate in M2. |
| FP16 precision loss in GPU GEMM/attention | Divergent text | FP32 accumulation (A8). Golden comparison gate in M3b. |
| PPA updater breakage by repo restructure | All Arda-Tech updates stop | §4.4 coexistence rule. M7 single-commit switch. |
| RAPL unreadable | No power number | A15 fallback chain. UI labels source. |
| 8B models barely fit | OOM / swap | Budget pre-flight refuses with itemized breakdown. Suggest smaller ctx/model. |

---

## 10. Milestone Report template (output at every GATE)

```markdown
## Milestone Mx Report
**Status:** COMPLETE | PARTIAL | BLOCKED (reason)
**Branch/commit:** rust-engine @ <sha>

### Files changed
- path: purpose (new/modified)

### New dependencies (with justification)
- crate x.y.z: why

### Commands run (verbatim) and results
$ cargo clippy --workspace --all-targets -- -D warnings   → <result>
$ cargo test --workspace                                   → <N passed / failed>
$ <milestone-specific commands>                            → <real output excerpt>

### Acceptance criteria
- [x] criterion (evidence)
- [ ] criterion (why not met / UNTESTED: reason)

### Measurements (real, with method)
| metric | value | how measured |

### Known limitations / deviations from plan
### Proposed next step
Awaiting `APPROVED Mx`.
```
