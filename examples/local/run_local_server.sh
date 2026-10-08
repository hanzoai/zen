#!/usr/bin/env bash
# run_local_server.sh - Run a local OpenAI-compatible inference server.
set -euo pipefail

MODEL_DIR="${HOME}/.cache/hanzo/models/qwen2.5-7b"
GGUF_FILE="${MODEL_DIR}/qwen2.5-7b-instruct-q4_k_m.gguf"
PORT=8080

echo "=== Starting local OpenAI-compatible server on :${PORT} ==="

if command -v llama-server &> /dev/null; then
  echo "Found llama-server. Starting server..."
  llama-server \
    -m "${GGUF_FILE}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --ctx-size 8192 \
    --n-gpu-layers 99
elif command -v vllm &> /dev/null; then
  echo "Found vLLM. Starting OpenAI-compatible API server..."
  vllm serve Qwen/Qwen2.5-7B-Instruct --port "${PORT}"
elif command -v ollama &> /dev/null; then
  echo "Found Ollama. Running qwen2.5:7b..."
  ollama run qwen2.5:7b
else
  echo "No local runtime detected. Install llama.cpp, vLLM, or Ollama, or point your client to evo.local:8730"
  exit 1
fi
