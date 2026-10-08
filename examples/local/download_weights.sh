#!/usr/bin/env bash
# download_weights.sh - Download open-weight models from Hugging Face for local offline execution.
set -euo pipefail

MODEL_DIR="${HOME}/.cache/hanzo/models"
mkdir -p "${MODEL_DIR}"

echo "=== Downloading open-weights to ${MODEL_DIR} ==="

# Example 1: Download GGUF quantized weights for CPU/Apple Silicon/Metal/ROCm
# Example model: Qwen2.5-Coder-7B-Instruct-GGUF or Llama-3.2-3B-Instruct-GGUF
if command -v huggingface-cli &> /dev/null; then
  echo "Using huggingface-cli..."
  huggingface-cli download \
    Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir "${MODEL_DIR}/qwen2.5-7b" \
    --local-dir-use-symlinks False
elif command -v curl &> /dev/null; then
  echo "Using curl to download GGUF direct..."
  mkdir -p "${MODEL_DIR}/qwen2.5-7b"
  curl -L -C - \
    -o "${MODEL_DIR}/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
fi

echo "Weights downloaded successfully to ${MODEL_DIR}/qwen2.5-7b/"
echo "Ready for local inference!"
