#!/usr/bin/env bash
# Generates golden reference tokenization and inference targets using llama.cpp.
set -euo pipefail

MODEL_PATH="${1:-}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: $0 <path-to-model.gguf>"
    echo "Requires a valid Llama-3 or Qwen2 GGUF model and llama.cpp binaries (llama-tokenize, llama-cli)."
    exit 1
fi

LLAMA_TOKENIZE="$(command -v llama-tokenize || true)"
LLAMA_CLI="$(command -v llama-cli || true)"

if [[ -z "${LLAMA_TOKENIZE}" || -z "${LLAMA_CLI}" ]]; then
    echo "Note: llama-tokenize and/or llama-cli not found on PATH."
    echo "To generate golden reference data, clone and compile llama.cpp:"
    echo "  git clone https://github.com/ggerganov/llama.cpp"
    echo "  cmake -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF"
    echo "  cmake --build llama.cpp/build --config Release -j\$(nproc)"
    echo "Then export PATH=\"\$PWD/llama.cpp/build/bin:\$PATH\""
    exit 0
fi

OUT_DIR="$(dirname "$0")/../tests/golden"
mkdir -p "${OUT_DIR}"

TEST_STRINGS=(
    "Hello world"
    "The quick brown fox jumps over the lazy dog."
    "Antigravity engine on Intel Gen8-10 hardware"
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    "1234567890 0xDEADBEEF 3.1415926535"
    "Special tokens: <|begin_of_text|> <|eot_id|>"
    "Unicode test: 你好世界! Bonjour le monde! こんにちは!"
    "Spaces:   three spaces, \t tab, \n newline"
    "Contractions: isn't won't shouldn't couldn't they're we've"
    "Mathematical symbols: ∫ ∑ ∏ √ ± ≠ ≤ ≥"
    "Path: /usr/local/bin/sulimo --profile=hybrid-gen9"
    "JSON: {\"model\": \"llama-3.2-3b\", \"kv_cache\": \"q8_0\"}"
    "Emoji: 🚀 🏎️ ⚡ 💻 🐧"
    "Whitespace trailing: hello    "
    "Whitespace leading:    world"
    "Repeated letters: aaaaaaaaaaaaaa bbbbbbbbbbbbb"
    "Punctuation run: !?!?!?!?!?!?!? ....."
    "URL: https://github.com/antux912012/Sulimo"
    "Markdown: # Súlimo **v0.2.0** `inline_code`"
    "Quotes: \"double quotes\" and 'single quotes' and `backticks`"
)

echo "==> Generating golden token IDs for 20 test strings..."
for i in "${!TEST_STRINGS[@]}"; do
    str="${TEST_STRINGS[$i]}"
    echo "String $i: ${str}"
    "${LLAMA_TOKENIZE}" -m "${MODEL_PATH}" "${str}" > "${OUT_DIR}/token_ids_${i}.txt" 2>/dev/null || true
done

PROMPTS=(
    "Once upon a time"
    "The capital of France is"
    "Write a short function in Python to add two numbers:"
    "Explain quantum computing in one simple sentence:"
    "The primary advantage of asymmetric CPU/GPU scheduling is"
)

echo "==> Generating 32-token greedy continuations for 5 reference prompts..."
for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    echo "Prompt $i: ${p}"
    "${LLAMA_CLI}" -m "${MODEL_PATH}" -p "${p}" -n 32 --temp 0.0 > "${OUT_DIR}/greedy_32_${i}.txt" 2>/dev/null || true
done

echo "==> Golden generation complete. Output saved to ${OUT_DIR}/"
