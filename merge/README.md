# ZENITH Architecture: Merging DeepSeek-V3 and Qwen3 using UMoE Technique

## Introduction

We present ZENITH (Zen Expert Network with Integrated Token Handling), a powerful merged language model that combines DeepSeek-V3 (Zen, 671B parameters with 37B activated) and Qwen3 (235B parameters with 22B activated) using a novel Unified Mixture-of-Experts (UMoE) technique. This merged architecture leverages the specialized capabilities of both models while maintaining computational efficiency through sparse activation of parameters.

ZENITH achieves remarkable performance across a wide range of tasks by combining DeepSeek-V3's exceptional code generation and technical explanation capabilities with Qwen3's mathematical reasoning and step-by-step thinking abilities. The model preserves Qwen3's dual-mode thinking capability, allowing users to explicitly request detailed reasoning using special tokens.

## Architecture Overview

### Foundational Architecture

ZENITH is built upon the sparse MoE architectures of both DeepSeek-V3 and Qwen3, where only a fraction of parameters activate for each token:
- DeepSeek-V3: 671B total parameters with 37B (5.5%) activated per token
- Qwen3: 235B total parameters with 22B (9.4%) activated per token
- Combined: ~900B total parameters with the same sparse activation pattern (activating ~37B parameters per token)

### Key Technical Innovations

#### 1. Unified Mixture-of-Experts (UMoE)

Our implementation employs the novel UMoE approach that unifies attention and feed-forward network (FFN) experts through a re-formulation of the attention mechanism:

- **Pre-mixing Attention**: Reformulates attention to reveal its underlying FFN-like structure, enabling unified expert architectures and parameter sharing
- **Unified Expert Design**: Implements standard two-layer FFNs as primary components for token processing and knowledge storage
- **Router Optimization**: Employs hidden-state initialization for router training to direct tokens to the most appropriate experts from both source models

#### 2. Tokenizer Unification

The model uses a unified tokenizer that preserves specialized vocabulary from both source models:
- **Union Approach**: Creates a combined vocabulary that preserves all tokens from both models
- **SLERP**: Uses Spherical Linear Interpolation for embedding vectors of tokens present in both models
- **Special Token Preservation**: Maintains critical functionality tokens, including Qwen3's `/think` and `/no_think` modes

#### 3. Dual-Mode Thinking Capability

ZENITH preserves Qwen3's distinctive dual-mode thinking capability through:
- **Tokenizer Integration**: Special tokens for invoking thinking modes are preserved in the merged vocabulary
- **Token-based Routing**: The router is trained to recognize these special tokens and activate appropriate experts
- **Prompt Template Compatibility**: The model maintains compatibility with Qwen3's prompt templates

## Performance and Capabilities

The ZENITH model combines specialized strengths from both source models:

### From DeepSeek-V3 (Zen)
- Advanced code generation and implementation
- Technical understanding and explanation
- System design and architecture
- Large context processing (128K tokens)

### From Qwen3
- Mathematical reasoning and problem-solving
- Step-by-step thinking and explanation
- Logical analysis and argument evaluation
- Explicit reasoning through thinking modes

## Deployment and Usage

### Hardware Requirements
- **GPU memory**: At least 90GB total (e.g., multiple A100/H100 GPUs)
- **Disk space**: ~2TB for both models and intermediate files
- **Compute**: Multiple high-end GPUs recommended

### Implementation
The repository provides a comprehensive framework for deploying ZENITH:
- Scripts for merging models using Mergekit
- Configuration files optimized for the UMoE technique
- Utilities for monitoring and validating the merged model
- Testing framework for evaluating performance

### Inference Support
ZENITH supports multiple inference frameworks:
- **DeepSeek-Infer**: Simple demo for both FP8 and BF16 inference
- **SGLang**: Full support with FP8 and BF16 precision
- **LMDeploy**: Efficient FP8 and BF16 inference
- **TensorRT-LLM**: BF16 inference with INT4/8 quantization
- **vLLM**: Tensor and pipeline parallelism support
- **LightLLM**: Efficient single-node or multi-node deployment

## Future Work

Our ongoing research focuses on:
1. **Performance optimization**: Enhancing the router mechanism for more efficient expert utilization
2. **Multi-token prediction**: Integrating MTP capabilities from both source models
3. **Advanced merging techniques**: Exploring MergeME for heterogeneous expert merging
4. **Evaluation benchmarks**: Developing specialized benchmarks to assess merged model capabilities
5. **Add more experts**: Building toward a comprehensive "Zen-Omni" model by incorporating additional expert models

## Acknowledgments

This implementation builds upon research from:
- The DeepSeek team for DeepSeek-V3 (Zen)
- The Qwen team for Qwen3
- The UMoE paper (Yang et al., 2025) for the unification technique
- The Mergekit team for MoE model merging tools

## Citation

```bibtex
@misc{hanzoai2024uaq,
  title={ZENITH: Zen Expert Network with Integrated Token Handling},
  author={Hanzo AI},
  year={2025},
  howpublished={https://github.com/hanzoai/zen}
}
```

```bibtex
@misc{yang2025umoe,
      title={UMoE: Unifying Attention and FFN with Shared Experts}, 
      author={Yuanhang Yang and Chaozheng Wang and Jing Li},
      year={2025},
      eprint={2505.07260},
      archivePrefix={arXiv}
}
```