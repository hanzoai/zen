#!/bin/bash
# Comprehensive utility script for model merge operations

function show_help() {
    echo "Utility script for DeepSeek-V3 and Qwen3 model merge operations"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  cleanup         - Remove temporary files and free up disk space"
    echo "  monitor         - Show GPU resource usage during merge"
    echo "  checkpoint      - Create a checkpoint of the current merge state"
    echo "  estimate        - Estimate disk space and memory requirements"
    echo "  recover [path]  - Recover from a checkpoint"
    echo "  validate        - Validate merged model structure"
    echo "  help            - Show this help message"
}

function cleanup() {
    echo "Cleaning up temporary files..."
    
    # Remove PyTorch cache files
    echo "Clearing PyTorch cache..."
    rm -rf ~/.cache/torch/hub
    rm -rf ~/.cache/torch/transformers
    
    # Clear CUDA cache
    echo "Clearing CUDA cache..."
    python -c "import torch; torch.cuda.empty_cache()"
    
    # Remove intermediate merge files if requested
    if [ "$1" == "--all" ]; then
        echo "WARNING: Removing ALL intermediate files..."
        read -p "This will delete merge progress. Continue? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "Removing intermediate merge files..."
            find ./merged_model -name "*_intermediate_*" -delete
            find ./merged_model -name "*.tmp" -delete
        fi
    fi
    
    # Clean up Python cache files
    echo "Cleaning Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    
    echo "Cleanup completed"
}

function monitor() {
    echo "Monitoring GPU and system resource usage..."
    
    interval=5
    if [ ! -z "$1" ]; then
        interval=$1
    fi
    
    # Check if tmux is available
    if command -v tmux &> /dev/null; then
        echo "Starting tmux monitoring session..."
        tmux new-session -d -s merge_monitor
        tmux split-window -h
        tmux select-pane -t 0
        tmux send-keys "watch -n $interval 'nvidia-smi'" C-m
        tmux select-pane -t 1
        tmux send-keys "watch -n $interval 'df -h; echo; free -h'" C-m
        tmux attach-session -t merge_monitor
    else
        echo "Tmux not available, using basic monitoring..."
        watch -n $interval "nvidia-smi; echo; df -h | grep -E '(Filesystem|merged_model)'; echo; free -h"
    fi
}

function checkpoint() {
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CHECKPOINT_DIR="./checkpoints/checkpoint_${TIMESTAMP}"
    
    echo "Creating checkpoint in ${CHECKPOINT_DIR}..."
    
    # Create checkpoint directory
    mkdir -p ${CHECKPOINT_DIR}
    
    # Copy configuration files
    cp config.yml ${CHECKPOINT_DIR}/
    
    # Save log files
    if [ -f "merge_process.log" ]; then
        cp merge_process.log ${CHECKPOINT_DIR}/
    fi
    
    # Archive intermediate files if they exist
    if [ -d "./merged_model" ]; then
        echo "Archiving merged model state..."
        
        # Create a manifest of files without copying the actual model weights
        echo "Creating file manifest..."
        find ./merged_model -type f -name "*.json" | xargs -I{} cp {} ${CHECKPOINT_DIR}/
        find ./merged_model -type f | sort > ${CHECKPOINT_DIR}/files_manifest.txt
        
        # Save checkpoint info
        echo "Checkpoint created at: $(date)" > ${CHECKPOINT_DIR}/checkpoint_info.txt
        echo "Original config: $(realpath config.yml)" >> ${CHECKPOINT_DIR}/checkpoint_info.txt
        echo "Model directory: $(realpath ./merged_model)" >> ${CHECKPOINT_DIR}/checkpoint_info.txt
    else
        echo "WARNING: No merged_model directory found. Checkpoint may be incomplete."
    fi
    
    echo "Checkpoint metadata created successfully at ${CHECKPOINT_DIR}"
    echo "NOTE: This checkpoint contains metadata only, not the full model weights."
}

function recover() {
    if [ -z "$1" ]; then
        echo "ERROR: Please specify a checkpoint directory"
        echo "Usage: $0 recover ./checkpoints/checkpoint_YYYYMMDD_HHMMSS"
        return 1
    fi
    
    CHECKPOINT_DIR=$1
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        return 1
    fi
    
    echo "Recovering from checkpoint: $CHECKPOINT_DIR"
    
    # Restore configuration file
    if [ -f "${CHECKPOINT_DIR}/config.yml" ]; then
        echo "Restoring configuration file..."
        cp ${CHECKPOINT_DIR}/config.yml ./config.yml.recovered
        echo "Configuration restored to ./config.yml.recovered"
    fi
    
    # Display checkpoint information
    if [ -f "${CHECKPOINT_DIR}/checkpoint_info.txt" ]; then
        echo "Checkpoint information:"
        cat ${CHECKPOINT_DIR}/checkpoint_info.txt
    fi
    
    echo "Recovery complete. Use the recovered configuration to continue the merge process."
}

function estimate() {
    echo "Estimating resource requirements for DeepSeek-V3 and Qwen3 merge..."
    
    # Check disk space
    echo "Checking available disk space..."
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    
    echo "Available disk space: ${AVAILABLE_SPACE}GB"
    echo "Estimated required disk space: ~2000GB (2TB)"
    
    if [ $AVAILABLE_SPACE -lt 2000 ]; then
        echo "WARNING: Available disk space may be insufficient"
    else
        echo "Disk space check: PASSED"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "\nChecking GPU resources..."
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{s+=$1} END {print s}')
        
        echo "GPU count: $GPU_COUNT"
        echo "Total GPU memory: ${GPU_MEMORY}MB"
        echo "Recommended GPU memory: 90000MB (90GB)"
        
        if [ $GPU_MEMORY -lt 90000 ]; then
            echo "WARNING: Total GPU memory may be insufficient for optimal performance"
        else
            echo "GPU memory check: PASSED"
        fi
    else
        echo "\nWARNING: No GPU detected. This merge requires GPUs."
    fi
    
    # Check system memory
    echo "\nChecking system memory..."
    SYS_MEMORY=$(free -m | awk '/^Mem:/{print $2}')
    
    echo "Available system memory: ${SYS_MEMORY}MB"
    echo "Recommended system memory: 64000MB (64GB)"
    
    if [ $SYS_MEMORY -lt 64000 ]; then
        echo "WARNING: System memory may be insufficient"
    else
        echo "System memory check: PASSED"
    fi
    
    echo "\nEstimated merge time: 6-48 hours (depending on hardware)"
}

function validate() {
    echo "Validating merged model structure..."
    
    if [ ! -d "./merged_model" ]; then
        echo "ERROR: Merged model directory not found"
        return 1
    fi
    
    # Check for essential files
    echo "Checking for essential files..."
    MISSING_FILES=0
    
    REQUIRED_FILES=("config.json" "tokenizer_config.json" "tokenizer.json")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "./merged_model/$file" ]; then
            echo "MISSING: $file"
            MISSING_FILES=$((MISSING_FILES+1))
        else
            echo "FOUND: $file"
        fi
    done
    
    # Check for model weights
    echo "\nChecking for model weights..."
    MODEL_FILES=$(find ./merged_model -name "*.safetensors" -o -name "*.bin" | wc -l)
    echo "Found $MODEL_FILES model weight files"
    
    if [ $MODEL_FILES -eq 0 ]; then
        echo "WARNING: No model weight files found"
        MISSING_FILES=$((MISSING_FILES+1))
    fi
    
    # Check for special tokens in tokenizer config
    if [ -f "./merged_model/tokenizer_config.json" ]; then
        echo "\nChecking for Qwen3 special tokens..."
        THINK_TOKEN=$(grep -c "/think" ./merged_model/tokenizer_config.json)
        
        if [ $THINK_TOKEN -eq 0 ]; then
            echo "WARNING: Qwen3 /think token not found in tokenizer config"
        else
            echo "FOUND: Qwen3 special tokens"
        fi
    fi
    
    # Summary
    echo "\nValidation summary:"
    if [ $MISSING_FILES -eq 0 ]; then
        echo "✅ Model structure appears valid"
    else
        echo "⚠️ Model has $MISSING_FILES issues that need attention"
    fi
}

# Main script logic
case "$1" in
    cleanup)
        cleanup $2
        ;;
    monitor)
        monitor $2
        ;;
    checkpoint)
        checkpoint
        ;;
    recover)
        recover "$2"
        ;;
    estimate)
        estimate
        ;;
    validate)
        validate
        ;;
    help|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
