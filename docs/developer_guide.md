# Zen Developer Guide: Technical Implementation

This guide provides developers with detailed information about Zen's architecture, router implementation, and integration approach for building spatially-aware AI applications.

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Router Implementation](#2-router-implementation)
3. [Expert Integration](#3-expert-integration)
4. [Spatial Processing](#4-spatial-processing)
5. [Deployment Guide](#5-deployment-guide)
6. [Extension Framework](#6-extension-framework)

## 1. Architecture Overview

Zen integrates multiple specialized models through a dynamic router architecture:

```
                            ┌────────────────────┐
                            │   Dynamic Router   │
                            └────────────────────┘
                                     │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
┌─────────▼──────────┐   ┌──────────▼──────────┐   ┌─────────▼──────────┐
│  LLaVA-NEXT-Inter. │   │ DeepSeek-V3/Qwen3   │   │       Phi-4        │
│   Vision System    │   │  Language Experts   │   │ Multimodal System  │
└─────────┬──────────┘   └──────────┬──────────┘   └─────────┬──────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Spatial Context   │
                         │      Manager        │
                         └─────────────────────┘
```

### Core Components

- **Dynamic Router**: Central orchestration component that directs computation to appropriate experts
- **Vision System**: Enhanced LLaVA-NEXT-Interleaved for visual understanding with spatial awareness
- **Language Experts**: DeepSeek-V3 (671B) and Qwen3 (235B) for specialized reasoning capabilities
- **Multimodal System**: Phi-4 with extensions for cross-modal understanding
- **Spatial Context Manager**: Maintains 3D scene representation and object relationships

### Communication Flow

The router manages bidirectional communication between all components:

1. Input analysis to determine task requirements and available modalities
2. Expert selection based on task classification and resource constraints
3. Parallel or sequential processing through selected experts
4. Integration of expert outputs into coherent response
5. Spatial context maintenance throughout processing pipeline

## 2. Router Implementation

The dynamic router is implemented as a trainable neural network that learns to efficiently allocate computation:

```python
class ZenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expert registry
        self.experts = load_experts(config.expert_paths)
        
        # Task classification head
        self.task_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.classifier_hidden),
            nn.GELU(),
            nn.Linear(config.classifier_hidden, config.num_task_classes)
        )
        
        # Expert selection policy
        self.selection_policy = nn.Sequential(
            nn.Linear(config.num_task_classes + config.resource_dims, 
                    config.policy_hidden),
            nn.GELU(),
            nn.Linear(config.policy_hidden, len(self.experts))
        )
        
        # Spatial context manager
        self.spatial_context = SpatialContextManager(config.spatial_config)
        
    def forward(self, inputs, resources=None):
        # Extract multimodal features
        features = self.extract_features(inputs)
        
        # Classify task type
        task_logits = self.task_classifier(features)
        task_probs = F.softmax(task_logits, dim=-1)
        
        # Prepare resource context if available
        if resources is not None:
            resource_embedding = self.embed_resources(resources)
            selection_context = torch.cat([task_probs, resource_embedding], dim=-1)
        else:
            selection_context = torch.cat([task_probs, 
                                          torch.zeros(task_probs.shape[0], 
                                                     self.config.resource_dims)], dim=-1)
        
        # Select experts based on task and resources
        selection_logits = self.selection_policy(selection_context)
        
        # For training: soft selection with straight-through estimator
        if self.training:
            selection_probs = F.softmax(selection_logits, dim=-1)
            selection = sample_gumbel_softmax(selection_logits, hard=True)
        # For inference: deterministic or stochastic selection
        else:
            if self.config.deterministic:
                selection = F.one_hot(selection_logits.argmax(dim=-1), 
                                    num_classes=len(self.experts))
            else:
                selection = sample_gumbel_softmax(selection_logits, 
                                                hard=True, 
                                                tau=self.config.temperature)
        
        # Process through selected experts (parallel implementation)
        outputs = torch.zeros_like(features)
        for i, expert in enumerate(self.experts):
            # Only process if this expert is selected
            expert_mask = selection[:, i].bool()
            if expert_mask.any():
                expert_inputs = self.prepare_expert_inputs(inputs, i)
                expert_outputs = expert(expert_inputs[expert_mask])
                outputs[expert_mask] = self.process_expert_outputs(expert_outputs, i)
        
        # Update spatial context with new information
        self.spatial_context.update(inputs, outputs)
        
        return outputs
```

### Router Training

The router is trained through a multi-objective approach:

1. **Task Performance**: Optimizing for accuracy on diverse benchmarks
2. **Resource Efficiency**: Minimizing computational overhead while maintaining performance
3. **Balanced Expert Utilization**: Preventing over-reliance on specific experts
4. **Spatial Coherence**: Maintaining consistent spatial understanding across processing stages

The training process uses Guided Router Policy Optimization (GRPO):

```python
def train_router_step(router, batch, optimizer, scheduler):
    # Forward pass with expert selection
    outputs = router(batch.inputs, resources=batch.resources)
    
    # Task-specific loss
    task_loss = compute_task_loss(outputs, batch.targets)
    
    # Resource efficiency loss
    efficiency_loss = compute_efficiency_loss(router.selection_probs, 
                                            batch.resources)
    
    # Expert utilization loss (encourage balanced use)
    utilization_loss = compute_utilization_loss(router.selection_probs)
    
    # Spatial coherence loss if applicable
    if batch.has_spatial:
        spatial_loss = compute_spatial_coherence_loss(
            router.spatial_context, batch.spatial_ground_truth)
    else:
        spatial_loss = 0.0
    
    # Combined loss with weighting
    total_loss = (
        task_loss * router.config.task_weight +
        efficiency_loss * router.config.efficiency_weight +
        utilization_loss * router.config.utilization_weight +
        spatial_loss * router.config.spatial_weight
    )
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    return {
        "task_loss": task_loss.item(),
        "efficiency_loss": efficiency_loss.item(),
        "utilization_loss": utilization_loss.item(),
        "spatial_loss": spatial_loss if isinstance(spatial_loss, float) 
                       else spatial_loss.item(),
        "total_loss": total_loss.item()
    }
```

## 3. Expert Integration

Zen integrates heterogeneous expert models through a standardized interface layer:

```python
class ExpertIntegration:
    def __init__(self, config):
        # Load expert models
        self.experts = {
            "vision": load_model("llava_next_interleaved", config.vision_path),
            "language": {
                "deepseek": load_model("deepseek_v3", config.deepseek_path),
                "qwen": load_model("qwen3", config.qwen_path)
            },
            "multimodal": load_model("phi4", config.phi4_path),
            "spatial": load_model("proprietary_3d", config.spatial_path)
        }
        
        # Create adapter layers for each expert
        self.adapters = {
            name: create_adapter(expert, config.hidden_size)
            for name, expert in self.flatten_experts().items()
        }
    
    def flatten_experts(self):
        """Convert nested expert dict to flat structure."""
        flat_experts = {}
        for key, value in self.experts.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_experts[f"{key}.{subkey}"] = subvalue
            else:
                flat_experts[key] = value
        return flat_experts
    
    def prepare_inputs(self, inputs, expert_name):
        """Convert universal inputs to expert-specific format."""
        if expert_name.startswith("vision"):
            return self._prepare_vision_inputs(inputs)
        elif expert_name.startswith("language"):
            return self._prepare_language_inputs(inputs)
        elif expert_name.startswith("multimodal"):
            return self._prepare_multimodal_inputs(inputs)
        elif expert_name.startswith("spatial"):
            return self._prepare_spatial_inputs(inputs)
        else:
            raise ValueError(f"Unknown expert: {expert_name}")
    
    def process_outputs(self, outputs, expert_name):
        """Convert expert-specific outputs to universal format."""
        adapter = self.adapters[expert_name]
        return adapter(outputs)
```

### Expert Specifications

#### LLaVA-NEXT-Interleaved (Vision)

The vision component is based on LLaVA-NEXT-Interleaved with specialized modifications:

```python
class SpatialVisionEncoder(nn.Module):
    def __init__(self, base_model_path, config):
        super().__init__()
        # Load base LLaVA-NEXT-Interleaved model
        self.base_model = load_llava_next(base_model_path)
        
        # Add spatial perception extensions
        self.depth_estimation = DepthEstimationHead(
            self.base_model.vision_encoder.output_dim, 
            config.depth_hidden_size
        )
        
        self.surface_normal = SurfaceNormalHead(
            self.base_model.vision_encoder.output_dim,
            config.normal_hidden_size
        )
        
        self.object_detection = ObjectDetectionHead(
            self.base_model.vision_encoder.output_dim,
            config.detection_hidden_size,
            config.num_object_classes
        )
        
        # Spatial feature integration
        self.spatial_fusion = SpatialFeatureFusion(
            config.vision_hidden_size,
            config.depth_hidden_size,
            config.normal_hidden_size,
            config.detection_hidden_size,
            config.fusion_hidden_size
        )
    
    def forward(self, images):
        # Base visual processing
        base_features = self.base_model.vision_encoder(images)
        
        # Spatial understanding components
        depth_features = self.depth_estimation(base_features)
        normal_features = self.surface_normal(base_features)
        detection_features = self.object_detection(base_features)
        
        # Integrate all spatial features
        fused_features = self.spatial_fusion(
            base_features, 
            depth_features, 
            normal_features, 
            detection_features
        )
        
        return fused_features
```

#### DeepSeek-V3 and Qwen3 (Language)

The language component dynamically routes between DeepSeek-V3 and Qwen3 experts:

```python
class DynamicLanguageExperts(nn.Module):
    def __init__(self, deepseek_path, qwen_path, config):
        super().__init__()
        # Load expert models
        self.deepseek = load_deepseek_v3(deepseek_path)
        self.qwen = load_qwen3(qwen_path)
        
        # Expert selection policy
        self.expert_selector = nn.Sequential(
            nn.Linear(config.hidden_size, config.selector_hidden),
            nn.GELU(),
            nn.Linear(config.selector_hidden, 2)  # DeepSeek or Qwen
        )
        
        # Output adaptation layers
        self.deepseek_adapter = nn.Linear(
            self.deepseek.output_dim, config.hidden_size)
        self.qwen_adapter = nn.Linear(
            self.qwen.output_dim, config.hidden_size)
    
    def forward(self, inputs, selection_strategy="dynamic"):
        # Extract features for expert selection
        features = self.extract_features(inputs)
        
        if selection_strategy == "dynamic":
            # Dynamic expert selection
            selection_logits = self.expert_selector(features)
            selection_probs = F.softmax(selection_logits, dim=-1)
            
            # Select expert based on probabilities
            if self.training:
                # During training use soft gumbel selection
                selection = sample_gumbel_softmax(selection_logits, hard=True)
            else:
                # During inference use deterministic or temperature sampling
                if getattr(self, "deterministic", True):
                    selection = F.one_hot(selection_logits.argmax(dim=-1), 
                                        num_classes=2)
                else:
                    selection = sample_gumbel_softmax(
                        selection_logits, 
                        hard=True, 
                        tau=getattr(self, "temperature", 1.0)
                    )
            
            # Process through selected expert
            outputs = torch.zeros(
                inputs.shape[0], self.config.sequence_length, self.config.hidden_size)
            
            # DeepSeek processing
            deepseek_mask = selection[:, 0].bool()
            if deepseek_mask.any():
                deepseek_outputs = self.deepseek(inputs[deepseek_mask])
                outputs[deepseek_mask] = self.deepseek_adapter(deepseek_outputs)
            
            # Qwen processing
            qwen_mask = selection[:, 1].bool()
            if qwen_mask.any():
                qwen_outputs = self.qwen(inputs[qwen_mask])
                outputs[qwen_mask] = self.qwen_adapter(qwen_outputs)
                
            return outputs
            
        elif selection_strategy == "ensemble":
            # Run both and ensemble results
            deepseek_outputs = self.deepseek(inputs)
            qwen_outputs = self.qwen(inputs)
            
            deepseek_adapted = self.deepseek_adapter(deepseek_outputs)
            qwen_adapted = self.qwen_adapter(qwen_outputs)
            
            # Compute ensemble weights
            selection_logits = self.expert_selector(features)
            weights = F.softmax(selection_logits, dim=-1)
            
            # Weighted average
            outputs = (weights[:, 0].unsqueeze(1).unsqueeze(2) * deepseek_adapted + 
                      weights[:, 1].unsqueeze(1).unsqueeze(2) * qwen_adapted)
            
            return outputs
            
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
```

#### Phi-4 Integration (Multimodal)

The multimodal component leverages Phi-4 with specialized spatial extensions:

```python
class SpatialMultimodalReasoning(nn.Module):
    def __init__(self, phi4_path, config):
        super().__init__()
        # Load base Phi-4 model
        self.phi4 = load_phi4(phi4_path)
        
        # Add spatial reasoning capabilities
        self.spatial_grounding = SpatialGroundingModule(
            self.phi4.hidden_size,
            config.grounding_hidden_size
        )
        
        # Spatial reference resolution
        self.reference_resolution = SpatialReferenceModule(
            self.phi4.hidden_size,
            config.reference_hidden_size
        )
        
        # Output adaptation
        self.output_adapter = nn.Linear(
            self.phi4.hidden_size + 
            config.grounding_hidden_size + 
            config.reference_hidden_size,
            config.hidden_size
        )
    
    def forward(self, text_inputs, visual_features, spatial_context):
        # Base Phi-4 processing
        phi4_outputs = self.phi4(text_inputs, visual_features)
        
        # Spatial reasoning extensions
        grounding_features = self.spatial_grounding(
            phi4_outputs, visual_features, spatial_context)
        
        reference_features = self.reference_resolution(
            phi4_outputs, spatial_context)
        
        # Integrate all features
        combined_features = torch.cat(
            [phi4_outputs, grounding_features, reference_features], 
            dim=-1
        )
        
        return self.output_adapter(combined_features)
```

## 4. Spatial Processing

Zen's spatial processing capability is implemented through a specialized framework:

```python
class SpatialContextManager:
    def __init__(self, config):
        self.config = config
        
        # Scene graph representation
        self.scene_graph = SceneGraph(config.scene_graph_config)
        
        # Physics simulation engine
        self.physics_engine = PhysicsEngine(config.physics_config)
        
        # Spatial memory
        self.spatial_memory = SpatialMemory(config.memory_config)
        
        # Object tracking system
        self.object_tracker = ObjectTracker(config.tracker_config)
    
    def initialize(self, visual_inputs=None, depth_inputs=None, 
                  sensor_inputs=None, text_description=None):
        """Initialize spatial context from inputs."""
        if visual_inputs is not None and depth_inputs is not None:
            # Initialize from visual + depth
            self._initialize_from_rgbd(visual_inputs, depth_inputs)
        elif sensor_inputs is not None:
            # Initialize from other sensor modalities
            self._initialize_from_sensors(sensor_inputs)
        elif text_description is not None:
            # Initialize from language description
            self._initialize_from_text(text_description)
        else:
            # Initialize empty context
            self._initialize_empty()
    
    def update(self, new_observations, actions=None):
        """Update spatial context with new observations."""
        # Update object tracking
        tracked_objects = self.object_tracker.update(new_observations)
        
        # Update scene graph
        self.scene_graph.update(tracked_objects)
        
        # Update physics predictions if actions provided
        if actions is not None:
            self.physics_engine.predict_next_state(self.scene_graph, actions)
        
        # Update spatial memory
        self.spatial_memory.integrate(self.scene_graph)
    
    def get_spatial_representation(self, format="graph"):
        """Get current spatial representation in specified format."""
        if format == "graph":
            return self.scene_graph.to_representation()
        elif format == "voxel":
            return self.scene_graph.to_voxel_grid()
        elif format == "point_cloud":
            return self.scene_graph.to_point_cloud()
        elif format == "language":
            return self.generate_spatial_description()
        else:
            raise ValueError(f"Unknown representation format: {format}")
    
    def generate_spatial_description(self):
        """Generate language description of spatial context."""
        # Implementation details...
        pass
```

### Scene Graph Implementation

The scene graph maintains spatial relationships between objects:

```python
class SceneGraph:
    def __init__(self, config):
        self.config = config
        self.nodes = {}  # Object nodes
        self.edges = {}  # Relationship edges
        
        # Spatial indexing for efficient queries
        self.spatial_index = SpatialIndex(config.index_type, config.index_params)
    
    def add_object(self, object_id, object_data):
        """Add object to scene graph."""
        self.nodes[object_id] = object_data
        self.spatial_index.insert(object_id, object_data.position, object_data.bounds)
        
        # Update relationships with existing objects
        self._update_relationships(object_id)
    
    def update_object(self, object_id, object_data):
        """Update existing object."""
        if object_id in self.nodes:
            old_data = self.nodes[object_id]
            self.nodes[object_id] = object_data
            
            # Update spatial index if position changed
            if not np.allclose(old_data.position, object_data.position) or \
               not np.allclose(old_data.bounds, object_data.bounds):
                self.spatial_index.remove(object_id)
                self.spatial_index.insert(
                    object_id, object_data.position, object_data.bounds)
            
            # Update relationships
            self._update_relationships(object_id)
        else:
            self.add_object(object_id, object_data)
    
    def remove_object(self, object_id):
        """Remove object from scene graph."""
        if object_id in self.nodes:
            # Remove from spatial index
            self.spatial_index.remove(object_id)
            
            # Remove all relationships involving this object
            edges_to_remove = []
            for edge_id, edge_data in self.edges.items():
                if edge_data.source == object_id or edge_data.target == object_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            
            # Remove node
            del self.nodes[object_id]
    
    def _update_relationships(self, object_id):
        """Update spatial relationships for given object."""
        obj = self.nodes[object_id]
        
        # Find nearby objects
        nearby_ids = self.spatial_index.query_radius(obj.position, self.config.relation_radius)
        
        # Compute relationships with each nearby object
        for other_id in nearby_ids:
            if other_id == object_id:
                continue
                
            other = self.nodes[other_id]
            
            # Compute relative position and orientation
            rel_pos = other.position - obj.position
            distance = np.linalg.norm(rel_pos)
            direction = rel_pos / distance if distance > 0 else np.zeros_like(rel_pos)
            
            # Determine relationship type
            rel_type = self._classify_relationship(obj, other, distance, direction)
            
            # Add or update edge
            edge_id = f"{object_id}_{other_id}"
            self.edges[edge_id] = RelationshipEdge(
                source=object_id,
                target=other_id,
                type=rel_type,
                distance=distance,
                direction=direction
            )
    
    def _classify_relationship(self, obj1, obj2, distance, direction):
        """Classify spatial relationship between objects."""
        # Implement relationship classification based on relative position,
        # orientation, size, and semantic properties
        # ...
        return relationship_type
    
    def to_representation(self):
        """Convert scene graph to tensor representation."""
        # Convert graph to tensor format suitable for model input
        # ...
        return representation
```

## 5. Deployment Guide

Zen supports multiple deployment configurations:

### Edge Deployment

Optimized for resource-constrained environments:

```python
def deploy_edge(config_path, output_path, target_size="0.8B"):
    """Deploy Zen to edge device configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Update for edge deployment
    edge_config = create_edge_config(config, target_size)
    
    # Load full model
    model = load_zen_model(config.model_path)
    
    # Optimize for edge
    edge_model = optimize_for_edge(model, edge_config)
    
    # Quantize model
    quantized_model = quantize_model(edge_model, edge_config.quantization)
    
    # Save optimized model
    save_model(quantized_model, output_path)
    
    print(f"Edge model deployed to {output_path}")
    print(f"Parameter count: {count_parameters(quantized_model)}")
    print(f"Memory footprint: {estimate_memory_footprint(quantized_model)}")
    print(f"Expected inference time: {estimate_inference_time(quantized_model)}")
```

### Enterprise Deployment

Balanced configuration for organizational environments:

```python
def deploy_enterprise(config_path, output_path, model_size="50B"):
    """Deploy Zen to enterprise environment."""
    # Load configuration
    config = load_config(config_path)
    
    # Update for enterprise deployment
    ent_config = create_enterprise_config(config, model_size)
    
    # Load full model
    model = load_zen_model(config.model_path)
    
    # Optimize for enterprise environment
    ent_model = optimize_for_enterprise(model, ent_config)
    
    # Create deployment package
    create_deployment_package(ent_model, ent_config, output_path)
    
    # Generate deployment documentation
    generate_documentation(ent_model, ent_config, output_path)
    
    print(f"Enterprise model deployed to {output_path}")
    print(f"Parameter count: {count_parameters(ent_model)}")
    print(f"Resource requirements:")
    print(f"  - RAM: {ent_config.ram_requirement}")
    print(f"  - GPU: {ent_config.gpu_requirement}")
    print(f"  - Storage: {ent_config.storage_requirement}")
```

### Cloud Deployment

Maximum capability through distributed infrastructure:

```python
def deploy_cloud(config_path, output_path, distributed=True, nodes=4):
    """Deploy Zen to cloud infrastructure."""
    # Load configuration
    config = load_config(config_path)
    
    # Update for cloud deployment
    cloud_config = create_cloud_config(config, distributed, nodes)
    
    # Load full model
    model = load_zen_model(config.model_path)
    
    # Prepare for distributed deployment if needed
    if distributed:
        model = prepare_distributed_model(model, cloud_config)
    
    # Create cloud deployment artifacts
    create_cloud_artifacts(model, cloud_config, output_path)
    
    # Generate deployment scripts
    generate_deployment_scripts(cloud_config, output_path)
    
    print(f"Cloud deployment package created at {output_path}")
    print(f"Deployment configuration:")
    print(f"  - Distributed: {distributed}")
    print(f"  - Nodes: {nodes}")
    print(f"  - Total parameters: {cloud_config.total_parameters}")
    print(f"  - Expected throughput: {cloud_config.expected_throughput}")
```

## 6. Extension Framework

Zen provides an extensible framework for adding new capabilities:

### Adding a New Expert

```python
def register_expert(name, model_path, config):
    """Register a new expert model with Zen."""
    # Load expert model
    expert_model = load_expert_model(model_path)
    
    # Create adapter layers
    input_adapter = create_input_adapter(expert_model, config)
    output_adapter = create_output_adapter(expert_model, config)
    
    # Register with expert registry
    expert_registry.register(
        name=name,
        model=expert_model,
        input_adapter=input_adapter,
        output_adapter=output_adapter,
        config=config
    )
    
    # Update router to include new expert
    update_router_for_expert(name, config)
    
    # Retrain router with new expert
    retrain_router(config.router_training)
    
    print(f"Expert '{name}' successfully registered and integrated")
```

### Adding a New Sensor Modality

```python
def add_sensor_modality(name, processor_path, config):
    """Add new sensor modality to Zen."""
    # Load sensor processor model
    processor = load_sensor_processor(processor_path)
    
    # Create adapter for spatial integration
    spatial_adapter = create_spatial_adapter(processor, config)
    
    # Register with sensor registry
    sensor_registry.register(
        name=name,
        processor=processor,
        spatial_adapter=spatial_adapter,
        config=config
    )
    
    # Update spatial context manager
    update_spatial_context_for_sensor(name, config)
    
    # Update router for new modality
    update_router_for_modality(name, config)
    
    print(f"Sensor modality '{name}' successfully added")
```

### Custom Application Development

```python
class ZenApplication:
    def __init__(self, config_path):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize Zen model
        self.model = load_zen_model(self.config.model_path)
        
        # Initialize application-specific components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize application-specific components."""
        # Implement in subclass
        pass
    
    def process_input(self, input_data):
        """Process application-specific input."""
        # Convert to model input format
        model_input = self.prepare_model_input(input_data)
        
        # Process through Zen model
        model_output = self.model(model_input)
        
        # Convert to application output format
        app_output = self.prepare_application_output(model_output)
        
        return app_output
    
    def prepare_model_input(self, input_data):
        """Convert application input to model input format."""
        # Implement in subclass
        pass
    
    def prepare_application_output(self, model_output):
        """Convert model output to application format."""
        # Implement in subclass
        pass
```

Example application implementation:

```python
class SpatialNavigationApp(ZenApplication):
    def initialize_components(self):
        # Initialize navigation-specific components
        self.map_manager = MapManager(self.config.map_config)
        self.path_planner = PathPlanner(self.config.planning_config)
        self.obstacle_detector = ObstacleDetector(self.config.obstacle_config)
    
    def prepare_model_input(self, input_data):
        # Extract sensor data
        visual_data = input_data.get('camera')
        depth_data = input_data.get('depth')
        lidar_data = input_data.get('lidar')
        imu_data = input_data.get('imu')
        
        # Current goal
        goal = input_data.get('goal')
        
        # Create model input dictionary
        model_input = {
            'visual': process_visual_data(visual_data),
            'depth': process_depth_data(depth_data),
            'lidar': process_lidar_data(lidar_data),
            'imu': process_imu_data(imu_data),
            'goal': encode_goal(goal),
            'map': self.map_manager.get_current_map()
        }
        
        return model_input
    
    def prepare_application_output(self, model_output):
        # Extract navigation plan from model output
        navigation_plan = extract_navigation_plan(model_output)
        
        # Refine with path planner
        refined_path = self.path_planner.refine_path(navigation_plan)
        
        # Check for obstacles
        safe_path = self.obstacle_detector.validate_path(refined_path)
        
        # Update map with new information
        self.map_manager.update(model_output.spatial_context)
        
        return {
            'path': safe_path,
            'map_updates': model_output.map_updates,
            'obstacle_alerts': model_output.obstacle_alerts,
            'next_waypoint': safe_path[0] if safe_path else None
        }
```

This developer guide provides a comprehensive overview of Zen's architecture, router implementation, and integration approach. By following these patterns, developers can extend Zen's capabilities and create spatially-aware AI applications for diverse domains.