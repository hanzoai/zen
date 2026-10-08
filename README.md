# Zen

Zen is Zen LM, the open model family from [Zoo Labs Foundation](https://zoo.ngo), a 501(c)(3)
non-profit. Open weights across language, code, vision, audio and retrieval: run them anywhere, or
call them on the Hanzo API.

- Family, model cards and papers: [zenlm/zen](https://github.com/zenlm/zen) · [zenlm.org](https://zenlm.org)
- Weights: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- Catalog and prices: [hanzo.ai/models/zen](https://hanzo.ai/models/zen)
- Docs: [docs.hanzo.ai/docs/models/zen](https://docs.hanzo.ai/docs/models/zen)

## Model Lineup

| SKU | Context | Description |
|---|---:|---|
| `zen5` (default) | 1M | Frontier open-weights ladder (glm-5.2 → deepseek-v4-pro) |
| `zen5-pro` | 1M | High-capacity reasoning |
| `zen5-coder` | 1M | Frontier code generation |
| `zen5-flash` | 64K | Ultra-low latency open-weights |
| `zen5-mini` | 32K | Lightweight local/edge deployment |
| `zen-vl` | 128K | Vision and multimodal understanding |
| `zen-embedding`| 8K | High-dimensional dense embeddings |

## Use Zen through the Hanzo API

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "zen5", "messages": [{"role": "user", "content": "Hello"}]}'
```

The API speaks the OpenAI wire format, so any OpenAI SDK works with
`base_url="https://api.hanzo.ai/v1"`. `GET https://api.hanzo.ai/v1/models` lists every Zen model
it serves.

## Runnable Examples

See [`examples/`](examples/README.md) for full runnable scripts and clients:

- **Local Execution (Offline Weights)**:
  - Download weights: [`examples/local/download_weights.sh`](examples/local/download_weights.sh)
  - Run local inference server: [`examples/local/run_local_server.sh`](examples/local/run_local_server.sh)
  - Python local client: [`examples/local/client_local.py`](examples/local/client_local.py)
  - Pure Rust local client: [`examples/local/client_local.rs`](examples/local/client_local.rs)
- **Cloud API**:
  - Python client: [`examples/cloud/client_cloud.py`](examples/cloud/client_cloud.py)
  - TypeScript client: [`examples/cloud/client_cloud.ts`](examples/cloud/client_cloud.ts)
  - cURL script: [`examples/cloud/client_cloud.sh`](examples/cloud/client_cloud.sh)
- **AI Gateway & Custom Org Routing**:
  - Per-org on-prem routing: [`examples/gateway/custom_org_routing.json`](examples/gateway/custom_org_routing.json)
