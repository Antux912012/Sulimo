# Súlimo (Arda's Breath): Model Hub & Multi-Architecture Expansion Build Plan
## Support for Qwen, Gemma, Liquid AI, and MiniCPM on Constrained Hardware (8 GB RAM / 30 GB Disk)

> **Document Version:** v1.0.0  
> **Target Audience:** Gemini 3.8 / Autonomous Coding Agent  
> **Reviewer/Gatekeeper:** Antonio (human approval required at each milestone gate)  
> **Repository:** `/home/antonio/Projects/Súlimo`  
> **Base Version:** Súlimo v0.2.0 (Rust Engine, GTK4 Libadwaita, Antigravity CLI)  
> **Target Version:** Súlimo v0.3.0  

---

## 0. Operating Rules for the Executing Agent

1. **Gate Discipline**: Implement exactly **one milestone at a time** (MM0 through MM5). Do not start the next milestone until the human explicitly replies `APPROVED MMx`.
2. **Hardware Constraints are Absolute**:
   - **Unified RAM Ceiling**: **5.20 GiB maximum** memory footprint (weights + KV cache + scratch arena + runtime). The host system has 8 GB RAM; exceeding 5.20 GiB triggers Linux OOM killer or swap thrashing.
   - **Disk Space Ceiling**: Total system disk is **30 GB SSD** with ~10–14 GB available for user files. Models must be compact (0.4 GB to 2.3 GB at Q4_0/Q4_K_M). The download manager **must verify free disk space via `statvfs` before starting any download** and reserve a 500 MiB safety buffer.
3. **Safety & Zero-Corrupt Downloads**:
   - Streaming HTTP(S) downloads with `Range` header resume support.
   - Downloads write to `<model>.part` with on-the-fly SHA-256 computation.
   - Atomic rename to `<model>.gguf` only after full completion and SHA-256 match.
   - Never shell out to external `curl`/`wget` binaries with unescaped strings. Use pure-Rust streaming HTTP (`reqwest` with `rustls` or `ureq`).
4. **Code Quality Requirements**:
   - `cargo fmt --all -- --check` must pass with 0 diffs.
   - `cargo clippy --workspace --all-targets -- -D warnings` must pass with 0 warnings.
   - `cargo test --workspace` must pass 100% of tests.
   - Every `unsafe` block must contain a `// SAFETY:` rationale.
   - Hot-path token generation must remain zero-allocation (0 bytes heap allocated during decode).

---

## 1. Curated Model Selection & Hardware Fit Analysis

### 1.1 Evaluated Candidates for 8 GB RAM / 30 GB Disk

| Model ID | Architecture | Parameter Count | Quantization | Download Size | RAM at 4K Context | 30 GB Disk Fit | Reasoning / Best Use Case |
|---|---|---|---|---|---|---|---|
| `qwen2.5-1.5b-instruct` | `qwen2` | 1.54B | Q4_0 | **0.98 GB** | **1.45 GiB** | **Optimal** | **Top General & Coding Workhorse**: Exceptional reasoning, instruction following, and multilingual support for its size. |
| `qwen2.5-0.5b-instruct` | `qwen2` | 0.49B | Q4_0 | **0.38 GB** | **0.65 GiB** | **Optimal** | **Ultra-Low Memory & High Speed**: Consumes almost no disk or RAM; decodes at >25 tok/s on legacy i5-7200U. |
| `gemma-2-2b-it` | `gemma2` | 2.61B | Q4_0 | **1.63 GB** | **2.25 GiB** | **Optimal** | **Top English & Conversational Quality**: Beats older 7B models on reasoning benchmarks; fits easily in budget. |
| `lfm-1.3b` (Liquid AI) | `lfm` / hybrid | 1.30B | Q4_0 | **0.82 GB** | **1.28 GiB** | **Optimal** | **Hybrid Conv/RNN/Attention**: Ultra-efficient memory footprint, fast TTFT, minimal KV cache scaling. |
| `minicpm3-4b-instruct` | `minicpm3` | 3.9B | Q4_0 | **2.32 GB** | **3.40 GiB** | **Viable (Large)** | **Maximum Intelligence Tier**: Near GPT-3.5 quality; uses 2.3 GB disk and 3.4 GB RAM, safely below 5.2 GB budget. |
| `llama-3.2-1b-instruct` | `llama` | 1.23B | Q4_0 | **0.78 GB** | **1.20 GiB** | **Optimal** | **Reference Baseline**: Compact, fast, well-tested on Súlimo engine. |
| `llama-3.2-3b-instruct` | `llama` | 3.21B | Q4_0 | **1.88 GB** | **2.70 GiB** | **Optimal** | **Reference 3B**: Solid general performance. |

### 1.2 Storage Footprint on 30 GB Disk
On a standard 30 GB Debian laptop installation:
- Linux Root OS + Desktop Environment: ~12 GB
- System swap / temp / cache: ~4 GB
- Available User Storage: **~14 GB**
- Storing 3 recommended models (`qwen2.5-1.5b` + `gemma-2-2b` + `lfm-1.3b`): **~3.43 GB total**
- Remaining disk space: **> 10.5 GB** (100% safe, no risk of disk exhaustion).

---

## 2. Architectural Extensions

### 2.1 Model Architecture Matrix in `sulimo-core`
Expand `ModelArch` enum and tensor mapping:

1. **`Qwen2` (`ModelArch::Qwen2`)**:
   - GQA attention with bias on Q, K, V projections (`blk.N.attn_q.bias`, etc.).
   - Standard RMSNorm (`eps = 1e-6`).
   - RoPE base `1,000,000`.
   - SwiGLU feed-forward network without bias.
2. **`Gemma2` (`ModelArch::Gemma2`)**:
   - Embedding normalization: weights scaled by `sqrt(hidden_dim)`.
   - Gemma RMSNorm: normalized activation scaled by `(1.0 + weight)`.
   - Logit soft-capping:
     - Attention logits: `score = tanh(score / 50.0) * 50.0`.
     - Output logits: `logits = tanh(logits / 30.0) * 30.0`.
   - Sliding window attention (SWA): alternating between local window (4096) and global attention.
   - GeGLU activation function in feed-forward layers.
3. **`MiniCPM3` (`ModelArch::MiniCPM3`)**:
   - Scale embedding multiplier (`scale_emb = 12.0`).
   - QK-Norm: RMSNorm applied to Q and K vectors prior to RoPE.
   - Tied word embeddings (output projection shares input token embedding weights).
4. **`LFM` (Liquid Foundation Models - `ModelArch::LFM`)**:
   - Hybrid sequence operator parsing (Conv1D state buffers + linear/GQA attention).
   - Recurrent state allocation within the static arena.

### 2.2 Model Manager Subsystem (`crates/sulimo-core/src/model_manager/`)
Create a dedicated model management module:
- **`registry.rs`**: Built-in curated catalog of tested models with verified direct Hugging Face GGUF URLs, file sizes, SHA-256 hashes, context lengths, and prompt templates (ChatML, Gemma, Llama-3, MiniCPM).
- **`downloader.rs`**: Resumable, chunked streaming downloader using pure Rust HTTP with:
  - `statvfs` pre-flight free disk check (`available >= size + 500 MiB`).
  - `Range: bytes=offset-` resumption support.
  - Streaming SHA-256 hasher (`sha2` crate).
  - Atomic rename (`.part` -> `.gguf`).
  - Thread-safe progress reporting channel (`bytes_downloaded`, `total_bytes`, `speed_bps`, `eta_seconds`).
  - Non-blocking cooperative cancellation token.
- **`store.rs`**: Local model repository scanner (`~/.local/share/sulimo/models/`), status detector (Installed, Downloading, Not Present), file validator, and deletion utility.

---

## 3. Milestone Execution Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MM0: Model Selection Grounding & Registry Specification                  │
│ • Curated model catalog with exact URLs, byte sizes, and SHA-256 hashes │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MM1: Architecture Expansion in `sulimo-core` & `sulimo-engine`          │
│ • Gguf parser & tensor loader for Gemma 2, Qwen2, MiniCPM, LFM          │
│ • Scalar and AVX2 kernels for Gemma RMSNorm, logit softcapping, QK-Norm │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MM2: Resumable Downloader & Model Manager Core                          │
│ • `statvfs` disk safety check, streaming HTTP, atomic rename, SHA256   │
│ • Local storage management in `~/.local/share/sulimo/models/`           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MM3: Antigravity CLI Model Hub & Slash Commands                         │
│ • `sulimo-cli models list/download/rm`                                   │
│ • REPL commands: `/models`, `/download <id>`, `/model <id>`             │
│ • Animated neon download progress card                                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MM4: Libadwaita Desktop GUI Model Hub                                   │
│ • GTK4 Model Hub window / navigation view                               │
│ • One-click async download with progress bar and cancel support        │
│ • Storage usage indicator (Models vs 30 GB Disk)                        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MM5: Integration Tests, End-to-End Verification & Release v0.3.0        │
│ • Live download test, inference verification, battery telemetry        │
│ • Debian package rebuild and PPA synchronization                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Milestone Details & Acceptance Criteria

### Milestone MM0: Model Selection Grounding & Registry Specification
- **Objective**: Finalize the curated model catalog with direct Hugging Face download endpoints, exact file sizes, SHA-256 hashes, context sizes, and template definitions.
- **Tasks**:
  1. Define `ModelMetadata` struct in `crates/sulimo-core/src/model_manager/mod.rs`.
  2. Implement `default_registry()` returning verified GGUF URLs:
     - `qwen2.5-1.5b-instruct-q4_0` (Qwen/Qwen2.5-1.5B-Instruct-GGUF)
     - `qwen2.5-0.5b-instruct-q4_0` (Qwen/Qwen2.5-0.5B-Instruct-GGUF)
     - `gemma-2-2b-it-q4_0` (bartowski/gemma-2-2b-it-GGUF)
     - `lfm-1.3b-q4_0` (LiquidAI/LFM-1.3B-GGUF or LiquidAI/LFM2.5-1.2B-GGUF)
     - `minicpm3-4b-instruct-q4_0` (openbmb/MiniCPM3-4B-GGUF)
     - `llama-3.2-1b-instruct-q4_0` (bartowski/Llama-3.2-1B-Instruct-GGUF)
  3. Validate that every model fits within the 5.20 GiB RAM budget and 30 GB disk constraint.
- **Verification**: Unit tests validating registry schema, non-empty URLs, and calculated memory footprints.

---

### Milestone MM1: Architecture Expansion in `sulimo-core` & `sulimo-engine`
- **Objective**: Extend GGUF loader and execution kernels to handle Gemma 2, Qwen2, MiniCPM, and LFM architectures.
- **Tasks**:
  1. Add `ModelArch::Gemma2`, `ModelArch::MiniCPM3`, and `ModelArch::LFM` to `sulimo-core::model_config`.
  2. Implement Gemma 2 forward pass adaptations:
     - Embedding scaling factor `sqrt(dim)`.
     - Gemma RMSNorm kernel (`x * (1.0 + w)`).
     - Logit soft-capping kernel (`tanh(x / cap) * cap`).
  3. Implement MiniCPM forward pass adaptations:
     - Embedding scale factor.
     - QK-Norm RMSNorm before RoPE.
  4. Ensure Qwen2 bias additions on QKV projection tensors are correctly applied during prefill and decode.
  5. Add unit tests for all new tensor parsing and kernel calculations against scalar reference implementations.
- **Verification**: `cargo test -p sulimo-core -p sulimo-cpu` passes with 100% precision equivalence.

---

### Milestone MM2: Resumable Downloader & Model Manager Core
- **Objective**: Build a robust, non-blocking HTTP streaming downloader with disk safety, SHA-256 verification, and resume capability.
- **Tasks**:
  1. Add `ureq` (or `reqwest` with `rustls`) and `sha2` dependencies to workspace.
  2. Implement `statvfs` pre-check: verify available disk space in download target partition >= `model_size + 500 MiB`.
  3. Implement chunked streaming download writing to `<model_id>.part`.
  4. Implement `Range: bytes=offset-` HTTP header handling to resume partially downloaded `.part` files.
  5. Compute streaming SHA-256 during download and verify against expected hash.
  6. Atomically rename `.part` to `.gguf` upon successful verification.
  7. Provide progress callback (`DownloadProgress { bytes_downloaded, total_bytes, speed_bps, eta_seconds }`) and cancellation flag (`Arc<AtomicBool>`).
- **Verification**: Integration test with mock HTTP server testing download, pause, resume, corrupt hash rejection, and disk full rejection.

---

### Milestone MM3: Antigravity CLI Model Hub & Slash Commands
- **Objective**: Provide a first-class terminal experience for browsing, downloading, and switching models in `sulimo-cli`.
- **Tasks**:
  1. Implement CLI subcommands:
     - `sulimo-cli models list`: Displays table of curated and installed models with sizes, RAM budgets, and download status.
     - `sulimo-cli models download <id>`: Downloads model with animated Antigravity progress bar.
     - `sulimo-cli models rm <id>`: Deletes model file safely.
     - `sulimo-cli models status`: Shows storage usage against disk limit.
  2. Implement interactive REPL slash commands:
     - `/models`: Displays models table inside active REPL session.
     - `/download <id>`: Downloads model with live progress in REPL.
     - `/model <id>`: Resolves `<id>` from downloaded models directory and loads it seamlessly.
  3. Ensure terminal output follows Antigravity visual language (electric cyan/neon accents, rounded box frames).
- **Verification**: Manual and automated tests verifying CLI subcommands and REPL slash commands.

---

### Milestone MM4: Libadwaita Desktop GUI Model Hub
- **Objective**: Implement a modern, intuitive Model Hub interface in `sulimo-gtk`.
- **Tasks**:
  1. Create `ModelHubDialog` or `ModelHubView` using Libadwaita widgets (`AdwPreferencesWindow`, `AdwActionRow`, `AdwStatusPage`).
  2. Display curated model cards with:
     - Model Name, Architecture badge, Download Size, RAM usage gauge.
     - Action button: "Download", "Cancel", "Load Model", or "Delete".
  3. Implement asynchronous download worker channel dispatching progress updates to the GTK main thread.
  4. Display live `GtkProgressBar`, download speed (MB/s), and ETA on the active model card.
  5. Add a "Local Storage" header bar gauge displaying: `Models: X.X GB / Free: YY.Y GB (30 GB Disk)`.
  6. Enable instant one-click switching to a downloaded model from the UI.
- **Verification**: Run `sulimo` GUI, open Model Hub, verify responsive rendering and non-blocking download UI.

---

### Milestone MM5: End-to-End Verification, Benchmarks & Release v0.3.0
- **Objective**: Full end-to-end testing, hardware benchmark verification, packaging, and PPA release.
- **Tasks**:
  1. Run full test suite: `./scripts/check.sh` (clippy, fmt, tests).
  2. Verify inference on downloaded models on Intel Core i5-7200U / HD 620:
     - Check RAM RSS remains strictly < 5.20 GiB.
     - Check iGPU prefill and AVX2 decode speed.
     - Check iGPU RC6 deep sleep residency > 85% during decode.
  3. Bump workspace version to `0.3.0` in `Cargo.toml`.
  4. Update `packaging/build_deb.sh` and build `sulimo_0.3.0_amd64.deb`.
  5. Update `README.md` and documentation with Model Hub instructions.
  6. Update Arda Tech PPA repository indices and push to GitHub.
- **Verification**: Automated test logs, clean benchmark output, and verified `.deb` package.

---

## 5. Milestone Completion Report Format

At the completion of each milestone, the agent must output:

```markdown
### Milestone MMx Report: [Milestone Title]
- **Status**: COMPLETED / READY FOR REVIEW
- **Code Changes**:
  - List of modified and created files
- **Verification Results**:
  - `cargo fmt --all -- --check`: PASSED
  - `cargo clippy --workspace --all-targets -- -D warnings`: PASSED
  - `cargo test --workspace`: PASSED (N tests passed)
- **Disk & Memory Audit**:
  - RAM overhead measured: X.XX GiB (Budget: 5.20 GiB)
  - Disk storage used: Y.YY GB (Limit: 30 GB SSD)
- **Screenshots / CLI Output Preview**:
  - Actual command outputs and UI state
- **Gate Approval Request**:
  - Ready for user approval (`APPROVED MMx`).
```
