# ZENITH Model Architecture and Implementation Guide

## 1. Project Overview

ZENITH (Zen Expert Network with Integrated Token Handling) is a revolutionary unified multimodal platform that combines multiple specialized AI experts through an innovative Mixture-of-Experts (UMoE) architecture. This advanced system integrates multiple model families—including Zen (671B parameters, 37B activated), Qwen3 (235B parameters, 22B activated), Zen-M for multimodal understanding, and Koe for voice synthesis—into a single, unified platform that scales from 22B to 900B+ parameters depending on deployment requirements.

### Key Features

- **Architecture**: Unified multimodal MoE architecture with flexible parameter scaling (22B-900B+)
- **Attention**: Revolutionary cross-modality attention with dynamic expert routing 
- **Expert Sharing**: Parameter-efficient design with shared experts across modalities
- **Tokenizer**: Unified vocabulary supporting text, voice, and multimodal inputs
- **Multimodal Capabilities**: Integration of Zen-M for UI understanding and Koe for voice synthesis
- **Sensor Extensibility**: Modular architecture designed for integrating additional sensors and modalities
- **Deployment Flexibility**: Dynamically scales from edge devices (22B) to high-performance servers (900B+)

## 2. Code Structure

```
/zen
├── figures/                # Model performance benchmark figures
├── inference/              # Inference implementation
│   ├── configs/            # Model configuration files
│   ├── convert.py          # Conversion utilities for model weights
│   ├── fp8_cast_bf16.py    # Utility to convert FP8 weights to BF16
│   ├── generate.py         # Text generation implementation
│   ├── kernel.py           # Low-level kernel implementations
│   ├── model.py            # Model architecture definition
│   └── requirements.txt    # Inference dependencies
├── merge/                  # ZENITH model merging implementation
│   ├── config.yml          # Configuration for Zen and Qwen3 merge
│   ├── merge_models.sh     # Main script for model merging process
│   ├── run_merge.sh        # Script to execute the merge operation
│   ├── setup_mergekit.sh   # Setup script for Mergekit installation
│   ├── test_merged_model.sh # Script to test the merged model
│   ├── update_config.py    # Script to update merged model configuration
│   ├── utils.sh            # Utility scripts for cleanup and monitoring
│   └── README.md           # Documentation for model merging process
└── README*.md              # Documentation files
```

## 3. Architecture Details

### 3.1 UMoE Architecture

The ZENITH model uses the Unified Mixture-of-Experts (UMoE) architecture that innovatively reformulates attention to reveal its underlying feed-forward network (FFN) structure:

- **Token Mixing**: Contextual information exchange through weighted summation of tokens
- **Expert Processing**: Standard two-layer FFNs as the primary components for knowledge storage
- **Router Mechanism**: Dynamic dispatch of tokens to the most relevant experts

This unified design allows attention and FFN layers to share parameters efficiently, with the distinction between them lying solely in how token mixing is handled.

#### Multi-head Attention Reformulation

The key insight in UMoE is the reformulation of the standard multi-head attention mechanism. Instead of the traditional formulation with separate value and output projections, the pre-mixing approach reorders operations to reveal an FFN-like structure:

```
# Traditional attention (simplified):
attention_output = softmax(QK^T/√d) × (V) × Wo

# UMoE pre-mixing attention (simplified):
attention_output = softmax(QK^T/√d) × X × (Wv × Wo)
```

This reformulation enables the `(Wv × Wo)` term to be implemented as a standard FFN (with appropriate non-linear activation), which can be shared with the actual FFN layers of the model.

### 3.2 Merged Model Configuration

The ZENITH model combines configurations from both source models:

- **Baseline Architecture**: Zen's core architecture with 61 transformer layers, 128 attention heads, and 7168 embedding dimension
- **Expert Configuration**: 256 routed experts with 8 activated per token
- **Router Strategy**: Hidden state initialization for optimal router training
- **Tokenizer**: Unified vocabulary with preserved special tokens

### 3.3 Key Components

#### Unified Expert Design

Experts in ZENITH are implemented as standard two-layer FFNs with:

- Small intermediate dimensions to align with Qwen3's architecture
- Non-linear activation functions between matrix multiplications
- Low-rank projections for query matrices to maintain parameter efficiency

#### Router Initialization

The router initialization uses the hidden state approach that:

- Creates representations from the last layer using carefully crafted prompts
- Guides the router to direct tokens to the most appropriate experts
- Balances Zen's code expertise with Qwen3's mathematical reasoning

#### Tokenizer Unification

The tokenizer unification strategy uses:

- **SLERP** (Spherical Linear Interpolation) for embedding vectors of tokens present in both models
- **Direct transfer** for tokens unique to each model
- **Token priority resolution** based on domain performance
- **Special token preservation** for critical functionality like Qwen3's thinking modes

## 4. Merging Methodology

### 4.1 Base Model Selection

Zen serves as the base model for attention and normalization layers, providing:

- Efficient Multi-head Latent Attention (MLA) mechanism
- Advanced positional encoding for long context
- Stable and well-optimized architecture

### 4.2 Expert Integration

Experts from both models are integrated through:

- **Expert Initialization**: Preservation of specialized weights from both models
- **Domain Specialization**: Careful selection of prompts to guide router initialization
- **Balancing**: Even distribution of experts to maintain performance across domains

### 4.3 Router Training

The router is trained using:

- **Specialized Prompts**: Examples that highlight each model's strengths
- **Hidden State Approach**: Most effective though computationally intensive strategy
- **Token Distribution Analysis**: Ensuring balanced activation across all experts

## 5. Implementation Patterns

### 5.1 Distributed Merging

The merge process leverages distributed computing through:

- **Multi-GPU Parallelism**: Utilizing tensor and pipeline parallelism
- **Memory Optimization**: Loading models in 8-bit precision during merging
- **Sharded Processing**: Breaking the merge into manageable chunks

### 5.2 Memory Management

Several techniques are used to manage the massive memory requirements:

- **Lazy Unpickling**: Loading model components only when needed
- **Gradient Accumulation**: Processing smaller batches to reduce memory footprint
- **Output Sharding**: Controlling shard size for optimal storage

### 5.3 Token Routing Optimization

The token routing process is optimized through:

- **Balanced Activation**: Ensuring even utilization of experts
- **Domain Awareness**: Using prompts that guide specialists toward appropriate domains
- **Controlled Redundancy**: Maintaining some redundancy for robustness

## 6. Inference

### 6.1 Deploying the Model

Multiple options are available for running inference:

1. **DeepSeek-Infer Demo**: Simple demo for both FP8 and BF16 inference
2. **SGLang**: Full support with FP8 and BF16 precision
3. **LMDeploy**: Efficient FP8 and BF16 inference
4. **TensorRT-LLM**: BF16 inference with INT4/8 quantization
5. **vLLM**: Tensor and pipeline parallelism support
6. **LightLLM**: Efficient single-node or multi-node deployment

### 6.2 Weight Conversion

The repository includes utilities for:

- Converting from mergekit format to inference-optimized format
- Converting from FP8 to BF16 weights for frameworks without FP8 support
- Handling special tokens and ensuring tokenizer compatibility

### 6.3 Generation Pipeline

The generation process:

1. Tokenize input prompt
2. Apply chat template if needed for conversation
3. Check for special tokens like `/think` and route appropriately
4. Forward pass through the model to get token probabilities
5. Temperature-controlled sampling for next token selection
6. Repeat until completion conditions are met

## 7. Development Considerations

When working with or extending the ZENITH model:

1. **Hardware Requirements**: The full model requires multiple high-end GPUs for inference
2. **Precision Options**: Consider using the native FP8 support for optimal performance
3. **Special Tokens**: Leverage the preserved special tokens like `/think` for enhanced capabilities
4. **Extended Context**: Take advantage of the 128K context support for long-document tasks

## 8. Integration with External Frameworks

The model can be integrated with:

- **vLLM** for high-throughput serving
- **SGLang** for both NVIDIA and AMD GPU support
- **TensorRT-LLM** for optimized inference on NVIDIA hardware
- **LMDeploy** for production-ready deployment
- **Huawei Ascend NPUs** through the MindIE framework

## 9. Future Work

Ongoing development focuses on:

1. **Performance optimization**: Enhancing the router mechanism for more efficient expert utilization
2. **Multi-token prediction**: Integrating MTP capabilities from both source models
3. **Advanced merging techniques**: Exploring MergeME for heterogeneous expert merging
4. **Evaluation benchmarks**: Developing specialized benchmarks to assess merged model capabilities
5. **Add more experts**: Building toward a comprehensive "Zen-Omni" model by incorporating additional expert models