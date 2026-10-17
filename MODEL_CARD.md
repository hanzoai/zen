---
language:
- en
- multilingual
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- foundation-model
- generative-ai
- agentic-coding
- open-weights
- zen
metrics:
- mmlu_pro
- humaneval
- swe_bench
model-index:
- name: zenlm/zen6
  results:
  - task:
      type: text-generation
      name: MMLU-Pro Reasoning
    metrics:
    - type: accuracy
      value: 0.812
      name: MMLU-Pro 5-shot
  - task:
      type: code-generation
      name: HumanEval
    metrics:
    - type: pass@1
      value: 0.894
      name: HumanEval Pass@1
  - task:
      type: software-engineering
      name: SWE-bench Verified
    metrics:
    - type: pass@1
      value: 0.448
      name: SWE-bench Resolved Rate
---

# Zen 6: Generative Intelligence Model Card

> 🌐 **Open Weights · Apache 2.0**  
> Developed by Zoo Labs Foundation and served globally on the Hanzo AI Gateway.  
> Weights: [`huggingface.co/zenlm`](https://huggingface.co/zenlm)

---

## 1. Lineup & Model Architecture

Zen 6 is an open-weight foundation model family optimized for agentic coding, deep reasoning, and long-context processing:

| Model ID | Parameters | Architecture | Context Window | Specialization |
|---|---|---|---|---|
| **`zen6`** | 70B MoE (8x) | Sparse Mixture of Experts | 131,072 | General reasoning, agentic planning, multimodality |
| **`zen6-flash`** | 8B Active | Ternary / Quantized Transformer | 65,536 | Ultra-efficient edge & cost-effective serving |
| **`zen6-coder`** | 32B Dense | Deep Attention Code Encoder-Decoder | 131,072 | Repository-scale code synthesis & automated debugging |
| **`zen-vl`** | 14B Dense | Multimodal Vision-Language | 32,768 | Visual document analysis, UI inspection, diagram reasoning |
| **`zen-embedding`** | 430M | 1024-dim Dense Vector Encoder | 8,192 | Semantic search, hybrid RAG, clustering |
| **`zen-reranker`** | 430M | Cross-Attention Document Reranker | 4,096 | Dense/hybrid RAG reranking and relevance scoring |

---

## 2. Verified Benchmark Results

| Benchmark Suite | Metric | Zen 6 | Llama 3.1 70B | Qwen 2.5 72B | DeepSeek-V2.5 |
|---|---|---|---|---|---|
| **MMLU-Pro** | 5-shot Accuracy | **81.2%** | 78.4% | 80.1% | 79.8% |
| **HumanEval** | Pass@1 (Python) | **89.4%** | 82.3% | 86.6% | 85.4% |
| **SWE-bench Verified** | Resolved Rate | **44.8%** | 38.2% | 41.5% | 42.1% |
| **MATH** | Exact Match | **76.5%** | 68.2% | 74.8% | 75.1% |

---

## 3. Quickstart with vLLM & Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen6"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")

prompt = "Write an asynchronous event loop in Rust using epoll."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4. Hub & Ecosystem Links

- **Weights Hub**: [`huggingface.co/zenlm`](https://huggingface.co/zenlm)
- **Documentation**: [docs.hanzo.ai/docs/models/zen](https://docs.hanzo.ai/docs/models/zen)
- **Playground**: [`huggingface.co/spaces/zenlm/zen-playground`](https://huggingface.co/spaces/zenlm/zen-playground)
