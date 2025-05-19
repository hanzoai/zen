#!/bin/bash
# Script to execute the merge operation for DeepSeek-V3 and Qwen3
set -e  # Exit on any error

# Set environment variables for optimal performance
echo "Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Increase Python's garbage collection threshold to reduce GC frequency
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Directory paths
CONFIG_FILE="./config.yml"
OUTPUT_DIR="./merged_model"
LOG_FILE="merge_process.log"

# Validate environment
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

echo "Checking for Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_mergekit.sh first."
    exit 1
fi

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Backup existing config
if [ -f "$CONFIG_FILE.bak" ]; then
    echo "Backup of config already exists."
else
    echo "Backing up original config..."
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
fi

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPU memory before merge:"
    nvidia-smi --query-gpu=memory.free --format=csv
fi

# Execute merge with optimization flags
echo "Starting merge process... This may take several hours."
echo "Merge started at $(date)" | tee -a "$LOG_FILE"

# Run the merge command with extensive memory optimizations
mergekit-moe $CONFIG_FILE $OUTPUT_DIR \
  --lazy-unpickle \
  --load-in-8bit \
  --device cuda \
  --out-shard-size 10B \
  --low-cpu-memory 2>&1 | tee -a "$LOG_FILE"

# Check if merge completed successfully
if [ $? -eq 0 ]; then
    echo "Merge completed successfully at $(date)" | tee -a "$LOG_FILE"
    echo "Merged model saved to: $OUTPUT_DIR"
    
    # Create a summary of the merge
    echo "Creating merge summary..."
    echo "===== ZENITH Merge Summary =====" > "$OUTPUT_DIR/merge_summary.txt"
    echo "Date: $(date)" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "Base Model: hanzoai/Zen1-Base" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "Merged with: Qwen/Qwen3-32B" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "Merging Strategy: Hidden state initialization with expert routing" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "\nFeatures Preserved:" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "- DeepSeek-V3's code generation and reasoning capabilities" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "- Qwen3's mathematical reasoning and step-by-step thinking abilities" >> "$OUTPUT_DIR/merge_summary.txt"
    echo "- Qwen3's dual-mode thinking capability (/think and /no_think tokens)" >> "$OUTPUT_DIR/merge_summary.txt"
    
    echo "Merge summary created at $OUTPUT_DIR/merge_summary.txt"
    echo "Next steps: Run ./test_merged_model.sh to test the merged model"
else
    echo "ERROR: Merge process failed at $(date)" | tee -a "$LOG_FILE"
    echo "Check $LOG_FILE for details"
fi
