<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# ZENITH: Hanzo AI's Unified Frontier Multimodal Platform

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Summary](#2-model-summary)
3. [Model Downloads](#3-model-downloads)
4. [Evaluation Results](#4-evaluation-results)
5. [Chat Website & API Platform](#5-chat-website--api-platform)
6. [How to Run Locally](#6-how-to-run-locally)
7. [License](#7-license)
8. [Citation](#8-citation)
9. [ZENITH: Unified Multimodal Platform](#9-zenith-unified-multimodal-platform)
10. [Decentralized AI Network](#10-decentralized-ai-network)
11. [Contact](#11-contact)


## 1. Introduction

We present Zen1, the cornerstone of Hanzo AI's groundbreaking family of Mixture-of-Experts (MoE) large language models and the foundation for ZENITH, our unified multimodal frontier platform. At its core, Zen1 features 671B total parameters with a highly efficient design that activates only 37B for each token, delivering breakthrough capabilities while maintaining computational efficiency.

As a Techstars-backed applied AI lab, Hanzo AI has pioneered revolutionary architectures including Multi-head Latent Attention (MLA) and our proprietary DeepSeekMoE framework. Zen1 introduces an innovative auxiliary-loss-free strategy for load balancing and multi-token prediction, establishing new benchmarks for performance and efficiency across multiple modalities.

Meticulously trained on 14.8 trillion diverse tokens, Zen1 delivers exceptional performance that rivals leading closed-source models while requiring just 2.788M H800 GPU hours for full training—a fraction of what comparable models demand. This training efficiency allows us to rapidly integrate and optimize new modality experts into our unified ZENITH platform.
<p align="center">
  <img width="80%" src="figures/benchmark.png">
</p>

## 2. Model Summary

---

**Architecture: Innovative Load Balancing Strategy and Training Objective**

- On top of the efficient architecture of Zen, we pioneer an auxiliary-loss-free strategy for load balancing, which minimizes the performance degradation that arises from encouraging load balancing.
-  We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model performance.
    It can also be used for speculative decoding for inference acceleration.

---

**Pre-Training: Towards Ultimate Training Efficiency**

- We design an FP8 mixed precision training framework and, for the first time, validate the feasibility and effectiveness of FP8 training on an extremely large-scale model.
- Through co-design of algorithms, frameworks, and hardware, we overcome the communication bottleneck in cross-node MoE training, nearly achieving full computation-communication overlap.
  This significantly enhances our training efficiency and reduces the training costs, enabling us to further scale up the model size without additional overhead.
- At an economical cost of only 2.664M H800 GPU hours, we complete the pre-training of Zen1 on 14.8T tokens, producing the currently strongest open-source base model. The subsequent training stages after pre-training require only 0.1M GPU hours.

---

**Post-Training: Knowledge Distillation from DeepSeek-R1**

-   We introduce an innovative methodology to distill reasoning capabilities from the long-Chain-of-Thought (CoT) model, specifically from one of the DeepSeek R1 series models, into standard LLMs, particularly Zen1. Our pipeline elegantly incorporates the verification and reflection patterns of R1 into Zen1 and notably improves its reasoning performance. Meanwhile, we also maintain a control over the output style and length of Zen1.

---


## 3. Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Zen1-Base | 671B | 37B | 128K   | [🤗 Hugging Face](https://huggingface.co/hanzoai/Zen1-Base)   |
| Zen1   | 671B | 37B |  128K   | [🤗 Hugging Face](https://huggingface.co/hanzoai/Zen1)   |

</div>

> [!NOTE]
> The total size of Zen1 models on Hugging Face is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights.

To ensure optimal performance and flexibility, we have partnered with open-source communities and hardware vendors to provide multiple ways to run the model locally. For step-by-step guidance, check out Section 6: [How_to Run_Locally](#6-how-to-run-locally).

For developers looking to dive deeper, we recommend exploring [README_WEIGHTS.md](./README_WEIGHTS.md) for details on the Main Model weights and the Multi-Token Prediction (MTP) Modules. Please note that MTP support is currently under active development within the community, and we welcome your contributions and feedback.

## 4. Evaluation Results
### Base Model
#### Standard Benchmarks

<div align="center">


|  | Benchmark (Metric) | # Shots | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | Zen1 |
|---|-------------------|----------|--------|-------------|---------------|---------|
| | Architecture | - | MoE | Dense | Dense | MoE |
| | # Activated Params | - | 21B | 72B | 405B | 37B |
| | # Total Params | - | 236B | 72B | 405B | 671B |
| English | Pile-test (BPB) | - | 0.606 | 0.638 | **0.542** | 0.548 |
| | BBH (EM) | 3-shot | 78.8 | 79.8 | 82.9 | **87.5** |
| | MMLU (Acc.) | 5-shot | 78.4 | 85.0 | 84.4 | **87.1** |
| | MMLU-Redux (Acc.) | 5-shot | 75.6 | 83.2 | 81.3 | **86.2** |
| | MMLU-Pro (Acc.) | 5-shot | 51.4 | 58.3 | 52.8 | **64.4** |
| | DROP (F1) | 3-shot | 80.4 | 80.6 | 86.0 | **89.0** |
| | ARC-Easy (Acc.) | 25-shot | 97.6 | 98.4 | 98.4 | **98.9** |
| | ARC-Challenge (Acc.) | 25-shot | 92.2 | 94.5 | **95.3** | **95.3** |
| | HellaSwag (Acc.) | 10-shot | 87.1 | 84.8 | **89.2** | 88.9 |
| | PIQA (Acc.) | 0-shot | 83.9 | 82.6 | **85.9** | 84.7 |
| | WinoGrande (Acc.) | 5-shot | **86.3** | 82.3 | 85.2 | 84.9 |
| | RACE-Middle (Acc.) | 5-shot | 73.1 | 68.1 | **74.2** | 67.1 |
| | RACE-High (Acc.) | 5-shot | 52.6 | 50.3 | **56.8** | 51.3 |
| | TriviaQA (EM) | 5-shot | 80.0 | 71.9 | 82.7 | **82.9** |
| | NaturalQuestions (EM) | 5-shot | 38.6 | 33.2 | **41.5** | 40.0 |
| | AGIEval (Acc.) | 0-shot | 57.5 | 75.8 | 60.6 | **79.6** |
| Code | HumanEval (Pass@1) | 0-shot | 43.3 | 53.0 | 54.9 | **65.2** |
| | MBPP (Pass@1) | 3-shot | 65.0 | 72.6 | 68.4 | **75.4** |
| | LiveCodeBench-Base (Pass@1) | 3-shot | 11.6 | 12.9 | 15.5 | **19.4** |
| | CRUXEval-I (Acc.) | 2-shot | 52.5 | 59.1 | 58.5 | **67.3** |
| | CRUXEval-O (Acc.) | 2-shot | 49.8 | 59.9 | 59.9 | **69.8** |
| Math | GSM8K (EM) | 8-shot | 81.6 | 88.3 | 83.5 | **89.3** |
| | MATH (EM) | 4-shot | 43.4 | 54.4 | 49.0 | **61.6** |
| | MGSM (EM) | 8-shot | 63.6 | 76.2 | 69.9 | **79.8** |
| | CMath (EM) | 3-shot | 78.7 | 84.5 | 77.3 | **90.7** |
| Chinese | CLUEWSC (EM) | 5-shot | 82.0 | 82.5 | **83.0** | 82.7 |
| | C-Eval (Acc.) | 5-shot | 81.4 | 89.2 | 72.5 | **90.1** |
| | CMMLU (Acc.) | 5-shot | 84.0 | **89.5** | 73.7 | 88.8 |
| | CMRC (EM) | 1-shot | **77.4** | 75.8 | 76.0 | 76.3 |
| | C3 (Acc.) | 0-shot | 77.4 | 76.7 | **79.7** | 78.6 |
| | CCPM (Acc.) | 0-shot | **93.0** | 88.5 | 78.6 | 92.0 |
| Multilingual | MMMLU-non-English (Acc.) | 5-shot | 64.0 | 74.8 | 73.8 | **79.4** |

</div>

> [!NOTE]
> Best results are shown in bold. Scores with a gap not exceeding 0.3 are considered to be at the same level. Zen1 achieves the best performance on most benchmarks, especially on math and code tasks.
> For more evaluation details, please check our paper.

#### Context Window
<p align="center">
  <img width="80%" src="figures/niah.png">
</p>

Evaluation results on the ``Needle In A Haystack`` (NIAH) tests.  Zen1 performs well across all context window lengths up to **128K**.

### Chat Model
#### Standard Benchmarks (Models larger than 67B)
<div align="center">

| | **Benchmark (Metric)** | **DeepSeek V2-0506** | **DeepSeek V2.5-0905** | **Qwen2.5 72B-Inst.** | **Llama3.1 405B-Inst.** | **Claude-3.5-Sonnet-1022** | **GPT-4o 0513** | **DeepSeek V3** |
|---|---------------------|---------------------|----------------------|---------------------|----------------------|---------------------------|----------------|----------------|
| | Architecture | MoE | MoE | Dense | Dense | - | - | MoE |
| | # Activated Params | 21B | 21B | 72B | 405B | - | - | 37B |
| | # Total Params | 236B | 236B | 72B | 405B | - | - | 671B |
| English | MMLU (EM) | 78.2 | 80.6 | 85.3 | **88.6** | **88.3** | 87.2 | **88.5** |
| | MMLU-Redux (EM) | 77.9 | 80.3 | 85.6 | 86.2 | **88.9** | 88.0 | **89.1** |
| | MMLU-Pro (EM) | 58.5 | 66.2 | 71.6 | 73.3 | **78.0** | 72.6 | 75.9 |
| | DROP (3-shot F1) | 83.0 | 87.8 | 76.7 | 88.7 | 88.3 | 83.7 | **91.6** |
| | IF-Eval (Prompt Strict) | 57.7 | 80.6 | 84.1 | 86.0 | **86.5** | 84.3 | 86.1 |
| | GPQA-Diamond (Pass@1) | 35.3 | 41.3 | 49.0 | 51.1 | **65.0** | 49.9 | 59.1 |
| | SimpleQA (Correct) | 9.0 | 10.2 | 9.1 | 17.1 | 28.4 | **38.2** | 24.9 |
| | FRAMES (Acc.) | 66.9 | 65.4 | 69.8 | 70.0 | 72.5 | **80.5** | 73.3 |
| | LongBench v2 (Acc.) | 31.6 | 35.4 | 39.4 | 36.1 | 41.0 | 48.1 | **48.7** |
| Code | HumanEval-Mul (Pass@1) | 69.3 | 77.4 | 77.3 | 77.2 | 81.7 | 80.5 | **82.6** |
| | LiveCodeBench (Pass@1-COT) | 18.8 | 29.2 | 31.1 | 28.4 | 36.3 | 33.4 | **40.5** |
| | LiveCodeBench (Pass@1) | 20.3 | 28.4 | 28.7 | 30.1 | 32.8 | 34.2 | **37.6** |
| | Codeforces (Percentile) | 17.5 | 35.6 | 24.8 | 25.3 | 20.3 | 23.6 | **51.6** |
| | SWE Verified (Resolved) | - | 22.6 | 23.8 | 24.5 | **50.8** | 38.8 | 42.0 |
| | Aider-Edit (Acc.) | 60.3 | 71.6 | 65.4 | 63.9 | **84.2** | 72.9 | 79.7 |
| | Aider-Polyglot (Acc.) | - | 18.2 | 7.6 | 5.8 | 45.3 | 16.0 | **49.6** |
| Math | AIME 2024 (Pass@1) | 4.6 | 16.7 | 23.3 | 23.3 | 16.0 | 9.3 | **39.2** |
| | MATH-500 (EM) | 56.3 | 74.7 | 80.0 | 73.8 | 78.3 | 74.6 | **90.2** |
| | CNMO 2024 (Pass@1) | 2.8 | 10.8 | 15.9 | 6.8 | 13.1 | 10.8 | **43.2** |
| Chinese | CLUEWSC (EM) | 89.9 | 90.4 | **91.4** | 84.7 | 85.4 | 87.9 | 90.9 |
| | C-Eval (EM) | 78.6 | 79.5 | 86.1 | 61.5 | 76.7 | 76.0 | **86.5** |
| | C-SimpleQA (Correct) | 48.5 | 54.1 | 48.4 | 50.4 | 51.3 | 59.3 | **64.8** |

</div>

> [!NOTE]
> All models are evaluated in a configuration that limits the output length to 8K. Benchmarks containing fewer than 1000 samples are tested multiple times using varying temperature settings to derive robust final results. Zen1 stands as the best-performing open-source model, and also exhibits competitive performance against frontier closed-source models.


####  Open Ended Generation Evaluation

<div align="center">



| Model | Arena-Hard | AlpacaEval 2.0 |
|-------|------------|----------------|
| DeepSeek-V2.5-0905 | 76.2 | 50.5 |
| Qwen2.5-72B-Instruct | 81.2 | 49.1 |
| LLaMA-3.1 405B | 69.3 | 40.5 |
| GPT-4o-0513 | 80.4 | 51.1 |
| Claude-Sonnet-3.5-1022 | 85.2 | 52.0 |
| Zen1 | **85.5** | **70.0** |

</div>

> [!NOTE]
> English open-ended conversation evaluations. For AlpacaEval 2.0, we use the length-controlled win rate as the metric.


## 5. Chat Website & API Platform
You can chat with Zen1 on DeepSeek's official website: [chat.hanzo.ai](https://chat.hanzo.ai/sign_in)

We also provide OpenAI-Compatible API at DeepSeek Platform: [platform.hanzo.ai](https://platform.hanzo.ai/)

## 6. How to Run Locally

Zen1 can be deployed locally using the following hardware and open-source community software:

1. **DeepSeek-Infer Demo**: We provide a simple and lightweight demo for FP8 and BF16 inference.
2. **SGLang**: Fully support the Zen1 model in both BF16 and FP8 inference modes, with Multi-Token Prediction [coming soon](https://github.com/sgl-project/sglang/issues/2591).
3. **LMDeploy**: Enables efficient FP8 and BF16 inference for local and cloud deployment.
4. **TensorRT-LLM**: Currently supports BF16 inference and INT4/8 quantization, with FP8 support coming soon.
5. **vLLM**: Support Zen1 model with FP8 and BF16 modes for tensor parallelism and pipeline parallelism.
6. **LightLLM**: Supports efficient single-node or multi-node deployment for FP8 and BF16.
7. **AMD GPU**: Enables running the Zen1 model on AMD GPUs via SGLang in both BF16 and FP8 modes.
8. **Huawei Ascend NPU**: Supports running Zen1 on Huawei Ascend devices.

Since FP8 training is natively adopted in our framework, we only provide FP8 weights. If you require BF16 weights for experimentation, you can use the provided conversion script to perform the transformation.

Here is an example of converting FP8 weights to BF16:

```shell
cd inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

> [!NOTE]
> Hugging Face's Transformers has not been directly supported yet.

### 6.1 Inference with DeepSeek-Infer Demo (example only)

#### System Requirements

> [!NOTE]
> Linux with Python 3.10 only. Mac and Windows are not supported.

Dependencies:
```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```
#### Model Weights & Demo Code Preparation

First, clone our Zen1 GitHub repository:

```shell
git clone https://github.com/hanzoai/Zen1.git
```

Navigate to the `inference` folder and install dependencies listed in `requirements.txt`. Easiest way is to use a package manager like `conda` or `uv` to create a new virtual environment and install the dependencies.

```shell
cd Zen1/inference
pip install -r requirements.txt
```

Download the model weights from Hugging Face, and put them into `/path/to/Zen1` folder.

#### Model Weights Conversion

Convert Hugging Face model weights to a specific format:

```shell
python convert.py --hf-ckpt-path /path/to/Zen1 --save-path /path/to/Zen1-Demo --n-experts 256 --model-parallel 16
```

#### Run

Then you can chat with Zen1:

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/Zen1-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

Or batch inference on a given file:

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/Zen1-Demo --config configs/config_671B.json --input-file $FILE
```

### 6.2 Inference with SGLang (recommended)

[SGLang](https://github.com/sgl-project/sglang) currently supports [MLA optimizations](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations), [DP Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models), FP8 (W8A8), FP8 KV Cache, and Torch Compile, delivering state-of-the-art latency and throughput performance among open-source frameworks.

Notably, [SGLang v0.4.1](https://github.com/sgl-project/sglang/releases/tag/v0.4.1) fully supports running Zen1 on both **NVIDIA and AMD GPUs**, making it a highly versatile and robust solution.

SGLang also supports [multi-node tensor parallelism](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208), enabling you to run this model on multiple network-connected machines.

Multi-Token Prediction (MTP) is in development, and progress can be tracked in the [optimization plan](https://github.com/sgl-project/sglang/issues/2591).

Here are the launch instructions from the SGLang team: https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

### 6.3 Inference with LMDeploy (recommended)
[LMDeploy](https://github.com/InternLM/lmdeploy), a flexible and high-performance inference and serving framework tailored for large language models, now supports Zen1. It offers both offline pipeline processing and online deployment capabilities, seamlessly integrating with PyTorch-based workflows.

For comprehensive step-by-step instructions on running Zen1 with LMDeploy, please refer to here: https://github.com/InternLM/lmdeploy/issues/2960


### 6.4 Inference with TRT-LLM (recommended)

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) now supports the Zen1 model, offering precision options such as BF16 and INT4/INT8 weight-only. Support for FP8 is currently in progress and will be released soon. You can access the custom branch of TRTLLM specifically for Zen1 support through the following link to experience the new features directly: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/deepseek_v3.


### 6.5 Inference with vLLM (recommended)

[vLLM](https://github.com/vllm-project/vllm) v0.6.6 supports Zen1 inference for FP8 and BF16 modes on both NVIDIA and AMD GPUs. Aside from standard techniques, vLLM offers _pipeline parallelism_ allowing you to run this model on multiple machines connected by networks. For detailed guidance, please refer to the [vLLM instructions](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). Please feel free to follow [the enhancement plan](https://github.com/vllm-project/vllm/issues/11539) as well.

### 6.6 Inference with LightLLM (recommended)

[LightLLM](https://github.com/ModelTC/lightllm/tree/main) v1.0.1 supports single-machine and multi-machine tensor parallel deployment for DeepSeek-R1 (FP8/BF16) and provides mixed-precision deployment, with more quantization modes continuously integrated. For more details, please refer to [LightLLM instructions](https://lightllm-en.readthedocs.io/en/latest/getting_started/quickstart.html). Additionally, LightLLM offers PD-disaggregation deployment for DeepSeek-V2, and the implementation of PD-disaggregation for Zen1 is in development.

### 6.7 Recommended Inference Functionality with AMD GPUs

In collaboration with the AMD team, we have achieved Day-One support for AMD GPUs using SGLang, with full compatibility for both FP8 and BF16 precision. For detailed guidance, please refer to the [SGLang instructions](#63-inference-with-lmdeploy-recommended).

### 6.8 Recommended Inference Functionality with Huawei Ascend NPUs
The [MindIE](https://www.hiascend.com/en/software/mindie) framework from the Huawei Ascend community has successfully adapted the BF16 version of Zen1. For step-by-step guidance on Ascend NPUs, please follow the [instructions here](https://modelers.cn/models/MindIE/deepseekv3).


## 7. License
This code repository is licensed under [the MIT License](LICENSE-CODE). The use of Zen1 Base/Chat models is subject to [the Model License](LICENSE-MODEL). Zen1 series (including Base and Chat) supports commercial use.

## 8. Citation
```
@misc{deepseekai2024deepseekv3technicalreport,
      title={Zen1 Technical Report},
      author={hanzoai},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## 9. ZENITH: Unified Multimodal Platform

We present ZENITH (Zen Expert Network with Integrated Token Handling), our unified multimodal frontier platform that combines multiple specialized experts within a flexible, parameter-efficient architecture. Through our innovative Unified Mixture-of-Experts (UMoE) technique, ZENITH creates a breakthrough system that scales from 22B to 900B+ parameters while dynamically activating only what's needed for each task.

This revolutionary architecture combines multiple specialized AI systems into a unified platform:

- **Zen's strengths**: Advanced code generation and explanation capabilities (671B parameters, 37B activated)
- **Qwen3's strengths**: Mathematical reasoning and step-by-step thinking (235B parameters, 22B activated)
- **Zen-M**: Multimodal understanding based on Phi-4, optimized for UI interaction on mobile and edge devices
- **Koe**: High-fidelity voice generation based on Nari Dia, fully integrated with the ZENITH platform
- **Parameter flexibility**: Dynamically scales from lightweight deployment (22B parameters) to full capability (900B+ parameters)

### Architecture Innovations

- **Pre-mixing Attention**: Revolutionary reformulation of attention mechanisms for optimal cross-modality routing
- **Expert Sharing**: Breakthrough parameter-efficient design with shared experts across modalities and architectures
- **Advanced Router Optimization**: Proprietary hidden state initialization for optimal task and modality routing
- **Multi-Model Integration**: Preserves and enhances specialized capabilities from all modality experts
- **Unified Multimodal Representation**: Common embedding space across text, voice, and UI understanding

### Multimodal Capabilities

- **Unified Text Understanding**: Industry-leading NLP with combined strengths of Zen and Qwen3
- **Multimodal Reasoning**: Zen-M enables contextual understanding of UI elements, images, and diagrams
- **Voice Synthesis**: Koe provides natural, expressive voice output seamlessly integrated with the platform
- **Adaptive UI Interaction**: Specialized for mobile and edge applications with contextual understanding of interface elements
- **Extensible Architecture**: Modular design for seamless integration of new sensors and modalities

### Implementation

The ZENITH unified multimodal platform is implemented using our proprietary routing and integration architecture. The `/merge` directory contains the core components for integrating new modality experts:

- Configuration files optimized for cross-modality expert integration
- Scaling scripts for varying deployment sizes from mobile (22B) to server (900B+)
- Evaluation framework for measuring cross-modal capabilities
- Adapter frameworks for integrating new sensors and data types

This modular architecture allows for seamless addition of new modalities, sensors, and specialized experts, creating a truly extensible AI platform that grows more capable over time.

For detailed implementation instructions and technical details, refer to the [merge directory README](/merge/README.md).

## 10. Decentralized AI Network

Hanzo AI, a Techstars-backed applied AI lab, is developing a revolutionary decentralized AI network to accelerate the training and deployment of our unified multimodal platform ZENITH.

### Network Overview

- **Distributed Compute Infrastructure**: Validator operators can contribute GPU resources to train and serve Hanzo AI's open-source multimodal experts
- **ZEN Token Incentives**: Earn ZEN tokens by hosting our models for training and inference across modalities
- **Democratized Access**: Making frontier multimodal AI accessible to everyone through shared resources
- **Accelerated Innovation**: Dramatically reducing training times for new modality experts and sensor integrations

### Benefits for Validators

- **Earn Rewards**: Generate sustainable income through ZEN token rewards
- **Support Open Science**: Contribute to cutting-edge open source multimodal AI research
- **Governance Rights**: Participate in governance of the Hanzo AI ecosystem
- **Early Access**: Gain priority access to new Hanzo AI modalities and capabilities

Our decentralized network represents a paradigm shift in how frontier multimodal AI systems are trained and deployed, creating a more equitable and efficient AI ecosystem that benefits everyone. Join us in democratizing access to the world's most advanced unified AI platform.

## 11. Contact
If you have any questions, please raise an issue or contact us at [service@hanzo.ai](service@hanzo.ai).
