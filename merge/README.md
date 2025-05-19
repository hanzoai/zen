# ZENITH Architecture: Merging Zen, Qwen3, Zen-M, and Koe into a Unified Multimodal Platform

## Introduction

We present ZENITH (Zen Expert Network with Integrated Token Handling), a revolutionary unified multimodal platform that combines Zen (671B parameters, 37B activated), Qwen3 (235B parameters, 22B activated), Zen-M for multimodal understanding, and Koe for voice synthesis—into a flexible, scalable system. Through our innovative Unified Mixture-of-Experts (UMoE) technique, ZENITH can scale from 22B to 900B+ parameters while maintaining exceptional computational efficiency through sparse expert activation.

ZENITH achieves unprecedented performance across multiple modalities by intelligently routing between specialized experts based on the specific task and input type. The platform preserves the specialized capabilities of all integrated systems while enabling new cross-modal interactions previously impossible with separate models.

## Architecture Overview

### Foundational Architecture

ZENITH is built upon a flexible, unified architecture that integrates multiple specialized models into a cohesive system:
- **Zen**: 671B total parameters with 37B (5.5%) activated per token
- **Qwen3**: 235B parameters with 22B (9.4%) activated per token
- **Zen-M**: Multimodal understanding based on Phi-4, optimized for UI interaction
- **Koe**: High-fidelity voice synthesis based on Nari Dia
- **Combined flexibility**: Scales from 22B parameters (mobile/edge) to 900B+ (server deployment)

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

The ZENITH platform unifies specialized capabilities from multiple expert systems:

### From Zen
- Advanced code generation and implementation
- Technical understanding and explanation
- System design and architecture
- Large context processing (128K tokens)

### From Qwen3
- Mathematical reasoning and problem-solving
- Step-by-step thinking and explanation
- Logical analysis and argument evaluation
- Explicit reasoning through thinking modes

### From Zen-M
- Multimodal understanding of UI elements
- Image and diagram comprehension
- Mobile and edge device optimization
- Context-aware interface interpretation

### From Koe
- High-fidelity voice synthesis
- Natural prosody and intonation
- Multi-lingual voice capabilities
- Voice style adaptation

## Deployment and Usage

### Hardware Requirements
- **Flexible deployment**: Scales from 22B parameters (mobile/edge) to 900B+ (server)
- **GPU memory**: 8GB for edge deployment, 90GB+ for full capability deployment
- **Storage**: Modular design allows for selective loading of needed experts
- **Compute**: Adapts to deployment environment from mobile devices to high-end servers

### Implementation
The repository provides a comprehensive framework for implementing the ZENITH multimodal platform:
- Modular expert integration framework for adding new modalities
- Configuration files for cross-expert routing optimization
- Adapters for different deployment environments (mobile to server)
- Multimodal integration utilities for sensor fusion
- Cross-modal training and evaluation tools

### Inference Support
ZENITH supports versatile deployment options across devices and environments:
- **Mobile/Edge Deployment**: Optimized for on-device inference with 22B-37B parameters
- **Server Deployment**: Full-capability inference with up to 900B+ parameters
- **Cross-Modal Inference**: Seamless processing of text, images, and voice inputs
- **Heterogeneous Hardware**: Optimization for various hardware (NVIDIA, AMD, mobile GPUs)
- **Sensor Integration**: Extensible architecture for incorporating additional sensors and data types
- **Custom Routing**: Configurable expert routing for specific application requirements

## Future Work

Our ongoing research focuses on:
1. **Expanding modalities**: Integrating additional specialized experts for new sensing modalities
2. **Cross-modal reasoning**: Enhancing interactions between different modality experts
3. **Deployment optimization**: Further reducing resource requirements for mobile/edge deployment
4. **Hardware acceleration**: Specialized implementations for different hardware architectures
5. **Sensor fusion**: Advanced techniques for combining inputs from multiple sensors
6. **On-device learning**: Enabling personalized fine-tuning across modalities on edge devices
7. **Community expansion**: Facilitating integration of community-contributed expert models

## Acknowledgments

This implementation builds upon research from:
- The Hanzo AI team for Zen and Zen-M
- The Qwen team for Qwen3
- The Nari Dia team for voice synthesis foundations in Koe
- The UMoE paper (Yang et al., 2025) for the unification technique
- The Phi-4 team for multimodal understanding foundations

## Citation

```bibtex
@misc{hanzoai2024uaq,
  title={ZENITH: Unified Multimodal Platform with Integrated Token Handling},
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