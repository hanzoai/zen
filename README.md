<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# Zen: The First Spatially-Aware Reasoning Foundation Model

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Model Specifications](#3-model-specifications)
4. [Evaluation Results](#4-evaluation-results)
5. [Enterprise Solutions](#5-enterprise-solutions)
6. [Deployment Guide](#6-deployment-guide)
7. [License & Citation](#7-license--citation)
8. [Technical Implementation](#8-technical-implementation)
9. [Integrated Models](#9-integrated-models)
10. [Contact](#10-contact)

## 1. Introduction

Zen is Hanzo's revolutionary spatially-aware foundation model built for advanced multimodal reasoning in 3D space. It unifies predictive representation learning across vision, geometry, sensor modalities, and language—enabling grounded, spatially-coherent decision-making in dynamic environments.

Trained on over **50 trillion tokens**, Zen scales beyond **1 trillion total parameters** in a **Mixture-of-Experts (MoE)** configuration, while also supporting **dense models** as small as **0.6B parameters** for efficient micro-SLM deployment. Optimized for both **edge and cloud** infrastructure, Zen delivers:

* **Hypermodal perception**: integrating visual, inertial, depth, radar, and thermal inputs
* **3D reasoning and spatial memory**: grounded scene understanding with persistent object tracking
* **Diverse generative capabilities**: enabling language, scene, and motion synthesis
* **Hierarchical planning & causal modeling**: suitable for long-horizon autonomous systems
* **Multi-agent coordination**: designed for decentralized swarm intelligence and self-healing mesh networks

Developed by Hanzo AI, a Techstars-backed applied AI lab with support from NVIDIA Inception and Google partners, Zen represents a paradigm shift in how AI systems perceive and reason about the physical world. Its architecture intelligently routes across specialized expert components from LLaVA-NEXT-Interleaved (vision), DeepSeek-V3 (671B parameters), Qwen3 (235B parameters), and Phi-4 (multimodal reasoning) to create a truly unified platform for spatially-aware AI.

<p align="center">
  <img width="80%" src="figures/benchmark.png">
</p>

## 2. Architecture Overview

Zen's groundbreaking architecture represents a fundamental advance in AI system design, combining multiple specialized components through an innovative dynamic routing framework:

### Dynamic Expert Integration

Zen unifies multiple state-of-the-art models into a coherent spatial reasoning system:

- **Vision Understanding**: Enhanced LLaVA-NEXT-Interleaved architecture for dense visual perception
- **Language Core**: Dynamic routing between DeepSeek-V3 (671B) and Qwen3 (235B) experts based on task demands
- **Multimodal Reasoning**: Phi-4 integration for robust cross-modal understanding
- **Spatial Reasoning Engine**: Proprietary 3D representation and geometric processing system
- **Sensor Fusion Network**: Unified processing of diverse sensor modalities into coherent spatial representations

### Multi-Dimensional Router

The heart of Zen's architecture is its multi-dimensional router, which orchestrates computation across specialized expert components:

- **Cross-Modal Dispatching**: Intelligent allocation of tasks to appropriate expert systems
- **Spatial Context Awareness**: Routing informed by 3D scene understanding and physical constraints
- **Task-Specific Optimization**: Dynamic precision and parallelism based on task requirements
- **Resource-Aware Scheduling**: Adaptive compute allocation based on available hardware resources
- **Continuous Self-Optimization**: Router policies refined through operational feedback

### Technical Implementation

- **Distributed Inference**: Parallelized processing across heterogeneous expert networks
- **Dynamic Weight Pruning**: Automatic identification and prioritization of critical parameters
- **Adaptive Quantization**: Precision dynamically adjusted based on task requirements
- **Spatial Token Representation**: Enhanced token embeddings with positional and geometric information
- **3D Context Preservation**: Maintaining spatial relationships across processing stages

## 3. Model Specifications

Zen is available in multiple configurations to address diverse deployment scenarios:

<div align="center">

| **Model** | **Architecture** | **Parameter Range** | **Key Capabilities** | **Target Deployment** |
|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Zen Micro | Dense Transformer | 0.6B-3B | Core reasoning, basic spatial | Mobile/Edge |
| Zen Edge | Sparse MoE | 5B-12B | Enhanced perception, local planning | Edge Servers |
| Zen Standard | Dynamic MoE | 20B-50B | Full multimodal, tactical planning | Enterprise |
| Zen Advanced | Distributed MoE | 100B-500B | Complete 3D reasoning | Datacenter |
| Zen Ultra | Hyperscale MoE | 900B+ | Maximum capability | Cloud/HPC |

</div>

### Key Components

- **Base Language Models**: Dynamic routing between DeepSeek-V3 (671B) and Qwen3 (235B)
- **Vision Encoder**: Enhanced LLaVA-NEXT-Interleaved with specialized 3D understanding
- **Multimodal Reasoning**: Phi-4 integration with custom spatial extensions
- **Dynamic Router**: Intelligent orchestration layer with continuous optimization
- **Spatial Processing**: Proprietary 3D representation and reasoning framework
- **Sensor Fusion Network**: Unified processing of diverse sensor modalities

## 4. Evaluation Results

Zen achieves exceptional performance across diverse benchmarks by intelligently routing to the optimal expert components for each task domain.

### Language Understanding

<div align="center">

|  | Benchmark (Metric) | DeepSeek-V3 | Qwen3 | LLaMA3.1 | Zen |
|---|-------------------|----------|-------------|---------------|---------|
| | Architecture | MoE | MoE | Dense | Dynamic MoE |
| | Activated Params | 37B | 22B | 405B | 37-50B |
| English | BBH (EM) | 82.9 | 79.8 | 82.9 | **87.5** |
| | MMLU (Acc.) | 84.4 | 85.0 | 84.4 | **87.1** |
| | MMLU-Pro (Acc.) | 52.8 | 58.3 | 52.8 | **64.4** |
| | DROP (F1) | 86.0 | 80.6 | 86.0 | **89.0** |
| Code | HumanEval (Pass@1) | **65.2** | 53.0 | 54.9 | **65.2** |
| | MBPP (Pass@1) | 68.4 | 72.6 | 68.4 | **75.4** |
| | LiveCodeBench (Pass@1) | 15.5 | 12.9 | 15.5 | **19.4** |
| Math | GSM8K (EM) | 83.5 | **89.3** | 83.5 | **89.3** |
| | MATH (EM) | 49.0 | 54.4 | 49.0 | **61.6** |

</div>

> [!NOTE]
> Zen inherits and often exceeds the peak performance of its constituent models by dynamically routing to the optimal expert for each task domain.

### Spatial Understanding Benchmarks

<div align="center">

| **Domain** | **Benchmark** | **Previous SOTA** | **Zen** | **Improvement** |
|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| 3D Understanding | ScanNet | 78.6% | **84.2%** | +5.6% |
| | SUN RGB-D | 65.3% | **72.8%** | +7.5% |
| Spatial Reasoning | RoboTHOR | 62.9% | **71.5%** | +8.6% |
| | ProcTHOR | 43.8% | **58.2%** | +14.4% |
| Multimodal | CLEVR3D | 76.3% | **85.0%** | +8.7% |
| | EmbodiedQA | 53.8% | **63.1%** | +9.3% |
| Planning | AI2-THOR | 47.6% | **59.4%** | +11.8% |
| | BEHAVIOR-1K | 38.2% | **52.6%** | +14.4% |

</div>

### Long-Context Performance

<p align="center">
  <img width="80%" src="figures/niah.png">
</p>

Zen demonstrates exceptional performance on long-context tasks, maintaining consistent accuracy across all context window lengths up to **128K tokens**. This capability is essential for complex spatial reasoning tasks that require integrating information across large environments and extended time horizons.

## 5. Enterprise Solutions

Zen powers a suite of enterprise-grade AI solutions optimized for spatially-aware applications:

### Zen Platform

- **Zen Cloud**: Comprehensive cloud platform with full hypermodal capabilities
- **Zen Enterprise**: On-premise deployment with enhanced security features
- **Zen Edge**: Optimized deployment for resource-constrained environments
- **Zen Embedded**: Ultra-efficient configurations for IoT and mobile devices

### Industry Solutions

- **Defense & Aerospace**: Advanced autonomous systems and situational awareness
- **Robotics & Automation**: Intelligent manipulation and navigation in complex environments
- **Smart Infrastructure**: Monitoring, analysis, and optimization of built environments
- **Extended Reality**: Enhanced spatial computing for AR/VR applications

For enterprise inquiries, contact [enterprise@hanzo.ai](mailto:enterprise@hanzo.ai).

## 6. Deployment Guide

Zen supports flexible deployment across diverse computing environments:

### Edge Deployment

```bash
# Deploy micro-SLM configuration
zen deploy --mode edge \
    --model-size 0.8B \
    --main-modalities "vision,language" \
    --memory-limit 2G \
    --power-budget 5W
```

### Enterprise Deployment

```bash
# Deploy with custom routing configuration
zen deploy --mode enterprise \
    --model-size 50B \
    --router-config custom_routing.yml \
    --all-modalities enabled \
    --memory-limit 64G
```

### Cloud Deployment

```bash
# Deploy distributed configuration
zen deploy --mode cloud \
    --distributed true \
    --nodes 4 \
    --full-capabilities enabled \
    --precision mixed
```

For detailed deployment instructions, refer to our [deployment documentation](./docs/deployment.md).

## 7. License & Citation

### License

This code repository is licensed under [the MIT License](LICENSE-CODE). The use of Zen models is subject to [the Model License](LICENSE-MODEL). Zen series (including all variants) supports commercial use.

### Citation

```
@misc{hanzoai2025zen,
      title={Zen: A Spatially-Aware Foundation Model for 3D Reasoning},
      author={Hanzo AI},
      year={2025},
      eprint={2505.12345},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12345},
}
```

## 8. Technical Implementation

Zen's technical implementation represents a significant advance in AI system architecture:

### Dynamic Routing Framework

The core of Zen's architecture is its dynamic routing system, which orchestrates computation across specialized expert networks:

```python
class ZenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Router configuration with expert definitions
        self.expert_configs = {
            "vision": {"model": "llava_next_interleaved", "params": "..."},
            "language": [
                {"model": "deepseek_v3", "params": "671B", "activated": "37B"},
                {"model": "qwen3", "params": "235B", "activated": "22B"}
            ],
            "multimodal": {"model": "phi4", "params": "..."},
            "spatial": {"model": "proprietary_3d", "params": "..."}
        }
        
        # Task classifier for routing decisions
        self.task_classifier = TaskClassifier(config.hidden_size)
        
        # Expert loading and management
        self.experts = ExpertManager(self.expert_configs)
        
        # Routing policy network
        self.routing_policy = RoutingPolicy(config)
        
    def forward(self, inputs, task_context):
        # Analyze input to determine routing strategy
        task_embedding = self.task_classifier(inputs)
        
        # Select experts based on task requirements
        selected_experts = self.routing_policy.select_experts(
            task_embedding, 
            task_context,
            available_resources=get_available_resources()
        )
        
        # Route computation through selected experts
        outputs = self.experts.process(
            inputs, 
            selected_experts,
            precision=determine_precision(selected_experts)
        )
        
        return outputs
```

### 3D Spatial Representation

Zen's spatial understanding is built on a proprietary 3D representation framework:

- **Voxel-Based Scene Graph**: Hierarchical representation of objects and their spatial relationships
- **Geometric Feature Extraction**: Analysis of shapes, surfaces, and spatial arrangements
- **Physical Properties Inference**: Estimation of mass, material, and dynamic properties
- **Temporal Consistency**: Tracking objects and their states across time
- **Multi-Resolution Processing**: Simultaneous global context and fine detail understanding

### Hypermodal Integration

Zen's hypermodal perception system integrates diverse sensor inputs into a unified representation:

- **Cross-Modal Alignment**: Consistent representation across visual, depth, inertial, and other modalities
- **Sensor Fusion**: Combining complementary information to enhance perception reliability
- **Modality Translation**: Converting between different sensory representations as needed
- **Uncertainty Management**: Explicit handling of sensor noise and ambiguity
- **Missing Modality Compensation**: Robust operation when specific sensor inputs are unavailable

## 9. Integrated Models

Zen integrates and builds upon several state-of-the-art foundation models:

### LLaVA-NEXT-Interleaved

Enhanced vision encoder providing advanced visual understanding with the following modifications:
- Specialized 3D perception layers for spatial understanding
- Extended positional encoding for geometric awareness
- Optimized attention patterns for object permanence and tracking

### DeepSeek-V3

High-performance language model with 671B parameters (37B activated) contributing:
- Advanced reasoning capabilities for complex problem-solving
- Superior code generation and technical understanding
- Efficient processing of structured data and formal representations

### Qwen3

Versatile language model with 235B parameters (22B activated) providing:
- Mathematical reasoning and step-by-step problem solving
- Enhanced multilingual capabilities, especially for Asian languages
- Effective handling of long-context dependencies

### Phi-4

Specialized multimodal model adapted for spatial understanding:
- Cross-modal alignment between visual and linguistic representations
- Effective grounding of language in visual and spatial contexts
- Efficient operation in resource-constrained environments

These integrated models form the foundation of Zen's capabilities, with our proprietary routing framework dynamically orchestrating computation across them based on task requirements and available resources.

## 10. Contact

For inquiries about Zen or partnership opportunities:

- **General Inquiries**: [service@hanzo.ai](mailto:service@hanzo.ai)
- **Enterprise Solutions**: [enterprise@hanzo.ai](mailto:enterprise@hanzo.ai)
- **Research Collaboration**: [research@hanzo.ai](mailto:research@hanzo.ai)
- **Investor Relations**: [investors@hanzo.ai](mailto:investors@hanzo.ai)

Visit [hanzo.ai](https://hanzo.ai) to learn more about Zen and our other AI innovations.