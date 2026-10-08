# Zen Examples: Local vs. Cloud Serving

Zen provides a unified OpenAI-compatible wire interface that can route to:
1. **Local Open-Weight Inference**: Run models on your own hardware (e.g. Apple Silicon, AMD Strix Halo, NVIDIA GPUs) with downloaded weights and zero external API keys.
2. **Hanzo Cloud**: Query managed frontier and open-weight models at `https://api.hanzo.ai/v1` with your `HANZO_API_KEY`.
3. **Hybrid AI Gateway**: Use Zen as your internal gateway configured via [platform.hanzo.ai](https://platform.hanzo.ai) to route per-org traffic between on-prem nodes and cloud fallbacks.

---

## Directory Structure

- [`local/`](./local): Scripts and clients for running open-weight models completely offline.
  - `download_weights.sh`: Download GGUF and SafeTensors weights from Hugging Face.
  - `run_local_server.sh`: Launch a local OpenAI-compatible server (vLLM, llama-server, Ollama).
  - `client_local.py`: Query local models with Python `openai` SDK.
  - `client_local.rs`: Pure Rust client querying local inference.
- [`cloud/`](./cloud): Scripts and clients querying managed Hanzo Cloud models.
  - `client_cloud.py`: Python SDK querying `api.hanzo.ai`.
  - `client_cloud.ts`: TypeScript SDK querying `api.hanzo.ai`.
  - `client_cloud.sh`: Direct cURL queries with API key auth.
- [`gateway/`](./gateway): Per-org routing configurations for the Hanzo AI Gateway.

---

## 1. Local Mode (Zero Cloud, Your Hardware)

### Step 1: Download weights
```bash
chmod +x local/download_weights.sh
./local/download_weights.sh
```

### Step 2: Run local inference server
```bash
chmod +x local/run_local_server.sh
./local/run_local_server.sh
```

### Step 3: Query locally
```bash
python local/client_local.py
```

---

## 2. Cloud Mode (Hanzo Cloud with API Key)

Set your Hanzo API key (from [platform.hanzo.ai](https://platform.hanzo.ai)):
```bash
export HANZO_API_KEY="your-api-key"
python cloud/client_cloud.py
```

---

## 3. What is `libcontrol`?

`libcontrol` is the pure-Rust C ABI library (`control` crate) that bridges Kai (the typed decision model) into the address space of Enso and Zen. 

Instead of making slow HTTP network hops for internal control decisions, Zen and Enso link `libcontrol` in-process:
- **Zero Network Overhead**: Sub-millisecond decision evaluation in native memory.
- **Topological DAG Decisions**: Evaluates deterministic policies and slot diffusion.
- **In-process Model Control**: Binds reasoning budgets, model routing, and tool escalation.
