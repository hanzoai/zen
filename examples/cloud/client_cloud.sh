#!/usr/bin/env bash
# client_cloud.sh - Query Hanzo Cloud API using cURL.
set -euo pipefail

if [[ -z "${HANZO_API_KEY:-}" ]]; then
  echo "Error: HANZO_API_KEY is not set" >&2
  echo "Set it with: export HANZO_API_KEY='your-key'" >&2
  exit 1
fi

BASE_URL="https://api.hanzo.ai/v1"

echo "=== Querying Hanzo Cloud (model: zen5) ==="
curl -s -X POST "${BASE_URL}/chat/completions" \
  -H "Authorization: Bearer ${HANZO_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen5",
    "messages": [
      {"role": "user", "content": "Explain hybrid local-cloud model serving in two sentences."}
    ],
    "temperature": 0.7
  }' | jq .
