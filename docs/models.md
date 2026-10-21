# Súlimo Curated Model Catalog & Architecture Guide

Súlimo provides a first-class curated model registry designed specifically for thin-and-light laptop hardware with **8 GB of system RAM** and a **256 GB host SSD**.

Every model in the registry is:
1. **Pre-verified against Súlimo's 5.20 GiB Unified Memory Ceiling**: The combined footprint of weights, 4K context circular KV cache, and scratch arenas is guaranteed to fit within 5.20 GiB, leaving at least ~2.8 GiB of RAM free for the Linux kernel, desktop environment, and browser.
2. **Pre-verified against the 30.0 GB SSD Model Storage Pool**: All 12 models combined occupy ~19.08 GB, allowing multiple architectures to be installed simultaneously within Súlimo's dedicated 30.0 GB model storage quota (`MAX_MODELS_STORAGE_BUDGET_BYTES = 30_000_000_000`).
3. **Cryptographically Guarded**: Every download endpoint is pinned to an exact SHA-256 hex digest. Downloads stream directly to disk with in-flight hashing and are atomically installed only when verification succeeds.

---

## 1. Curated Models Registry Table

| Model ID | Architecture | Quantization | Nominal Params | Download Size | 4K RAM Footprint | Context Window | Recommended Use Case |
|---|---|---|---|---|---|---|---|
| `qwen2.5-3b-instruct` | `Qwen2` | `Q4_K_M` | 3.09B | 2.10 GB | 2.75 GiB | 32,768 | Premier 3B model with exceptional reasoning, math, and code capabilities |
| `minicpm3-4b` | `MiniCPM3` | `Q4_K_M` | 3.90B | 2.47 GB | 3.40 GiB | 32,768 | Near GPT-3.5 performance in a 2.47 GB footprint with 32K context |
| `llama-3.2-3b-instruct` | `Llama` | `Q4_K_M` | 3.21B | 2.02 GB | 2.70 GiB | 131,072 | Meta's flagship edge 3B instruct model with strong multi-turn reasoning |
| `lfm2.5-2.6b` | `LFM` | `Q4_0` | 2.60B | 1.59 GB | 2.10 GiB | 32,768 | Hybrid Conv/RNN/Attention foundation model from Liquid AI |
| `gemma-2-2b-it` | `Gemma2` | `Q4_K_M` | 2.61B | 1.71 GB | 2.25 GiB | 8,192 | Google Gemma 2 with dual RMSNorm and attention logit soft-capping |
| `qwen2.5-1.5b-instruct` | `Qwen2` | `Q4_0` | 1.54B | 1.07 GB | 1.45 GiB | 32,768 | Top coding and reasoning workhorse in a compact ~1 GB footprint |
| `minicpm5-2b` | `MiniCPM3` | `Q4_K_M` | 2.40B | 1.56 GB | 2.10 GiB | 32,768 | High-capability 2B bilingual model with strong reasoning from OpenBMB |
| `minicpm5-1b` | `MiniCPM3` | `Q4_K_M` | 1.10B | 688 MB | 1.10 GiB | 32,768 | OpenBMB lightweight on-device bilingual powerhouse model |
| `llama-3.2-1b-instruct` | `Llama` | `Q4_0` | 1.23B | 773 MB | 1.20 GiB | 131,072 | Meta reference lightweight instruct model, fast and general-purpose |
| `qwen2.5-0.5b-instruct` | `Qwen2` | `Q4_0` | 0.49B | 429 MB | 0.65 GiB | 32,768 | Ultra-low resource edge model with blazing fast inference (>25 tok/s) |
| `lfm2.5-230m` | `LFM` | `Q4_0` | 0.23B | 149 MB | 0.25 GiB | 32,768 | Ultra-compact sub-150 MB liquid model for battery-first edge tasks |
| `qwen2.5-1.5b-instruct-q8` | `Qwen2` | `Q8_0` | 1.54B | 1.89 GB | 2.22 GiB | 32,768 | Full Q8_0 precision weights with virtually zero quantization loss |

---

## 2. Supported Architecture Specifications

### Qwen 2.5 (`ModelArch::Qwen2`)
- **Key Features**: Grouped-Query Attention (GQA), RoPE positional embeddings, QKV linear projection biases, and SwiGLU activation.
- **Chat Template**: ChatML format (`<|im_start|>role\ncontent<|im_end|>`).
- **Prompting Strength**: Outstanding coding, mathematical problem solving, structured JSON generation, and concise instruction following.

### Gemma 2 (`ModelArch::Gemma2`)
- **Key Features**:
  - **Dual RMSNorm**: Uses standard RMSNorm with `(1 + w)` scaling (`rmsnorm_gemma_scalar` and `rmsnorm_gemma_avx2`).
  - **Attention & Logit Soft-Capping**: Prevents logits from exploding into extreme values without clipping gradients:
    $$\text{softcap}(x, C) = C \cdot \tanh\left(\frac{x}{C}\right)$$
    Standard cap values: $C_{\text{attn}} = 50.0$, $C_{\text{final}} = 30.0$.
  - **GeGLU Activation**: Gated GELU non-linearities for feed-forward network layers.
  - **Post-Attention & Post-FFN Normalization**: Stabilizes multi-layer gradient flows.
- **Chat Template**: Gemma format (`<start_of_turn>role\ncontent<end_of_turn>`).
- **Prompting Strength**: High conversational quality, academic writing, and encyclopedic knowledge.

### Liquid Foundation Models (`ModelArch::LFM`)
- **Key Features**: Liquid AI's proprietary hybrid dynamical system architecture. Combines continuous-time convolutional and recurrent token mixing with selective attention for unmatched memory efficiency.
- **Chat Template**: ChatML format.
- **Prompting Strength**: Unrivaled inference speed and battery longevity on low-power Intel hardware, making `lfm2.5-230m` and `lfm2.5-2.6b` ideal for background assistants.

### MiniCPM 3 & 5 (`ModelArch::MiniCPM3`)
- **Key Features**:
  - **QK-Normalization**: Applies RMSNorm directly to Query and Key projections prior to RoPE to prevent attention entropy collapse.
  - **Embedding Scaling**: Scales initial token embeddings by $\sqrt{d_{\text{model}}}$.
- **Chat Template**: ChatML format.
- **Prompting Strength**: Strong bilingual capabilities (English/Chinese), document parsing, and high reasoning density per parameter.

### Llama 3.2 (`ModelArch::Llama`)
- **Key Features**: Grouped-Query Attention (GQA), base RoPE frequency scaling ($\theta = 500{,}000$), SwiGLU activations, and extended 128K context window capabilities.
- **Chat Template**: Llama 3 format (`<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`).
- **Prompting Strength**: General knowledge, roleplay, creative writing, and summarization.

---

## 3. Storage Pool & Memory Constraints

### 30.0 GB SSD Model Storage Pool
- Default Directory: `~/.local/share/sulimo/models` (override via `export SULIMO_MODELS_DIR=/custom/path`).
- Maximum Storage Budget: `MAX_MODELS_STORAGE_BUDGET_BYTES = 30_000_000_000` (30.0 GB).
- Safety Headroom: Súlimo queries filesystem free space via POSIX `statvfs` and mandates at least `file_size + 500 MiB` free disk space before initiating any download.

### 5.20 GiB Unified Memory Ceiling
- Enforces safe execution on an **8 GB shared RAM host**:
  $$\text{RAM}_{\text{total}} = \text{RAM}_{\text{weights}} + \text{RAM}_{\text{KV-cache}} + \text{RAM}_{\text{scratch}} \le 5.20\text{ GiB}$$
- At 4K context, all 12 curated models fit safely with 1.8 to 4.5 GiB of system RAM left over for the OS and desktop applications.

---

## 4. CLI Model Management Commands

```bash
# List all curated models, download status, and storage pool usage
sulimo-cli models list

# Output machine-readable JSON summary of storage and models
sulimo-cli models list --json

# Search curated registry by keyword, architecture, or family
sulimo-cli models search qwen
sulimo-cli models search liquid

# Download a curated model with resumable streaming progress
sulimo-cli models download qwen2.5-1.5b-instruct

# Verify cryptographic SHA-256 integrity of an installed model
sulimo-cli models verify qwen2.5-1.5b-instruct

# Remove an installed model and reclaim SSD storage
sulimo-cli models remove qwen2.5-1.5b-instruct

# Run single-prompt inference (auto-prompts to download if not yet installed)
sulimo-cli run qwen2.5-1.5b-instruct -p "Explain gravity in simple terms."

# Launch interactive chat REPL with a model
sulimo-cli chat qwen2.5-1.5b-instruct
```

---

## 5. Adding Custom GGUF Models

To use external models not included in the curated registry:
1. Place any GGUF file in `~/.local/share/sulimo/models/` (e.g. `my-custom-model.gguf`).
2. Súlimo automatically detects the file in both CLI (`sulimo-cli models list`) and Desktop GUI.
3. Run inference directly:
   ```bash
   sulimo-cli run my-custom-model.gguf -p "Hello!"
   ```
