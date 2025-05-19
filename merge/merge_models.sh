#!/bin/bash
# Main script for DeepSeek-V3 and Qwen3 model merge

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Directory paths
MERGED_MODEL_DIR="./merged_model"
INFERENCE_DIR="./merged_model_inference"

# Function to display start of step
function step() {
    echo -e "\n\033[1;36m==== $1 ====\033[0m"
}

# Function to display error and exit
function error() {
    echo -e "\033[1;31mERROR: $1\033[0m"
    exit 1
}

# Setup environment
step "Setting up environment"
chmod +x setup_mergekit.sh
./setup_mergekit.sh || error "Failed to set up environment"

# Check if model paths are available and valid
step "Checking model availability"
if [ ! -d "hanzoai/Zen1-Base" ]; then
    echo "Warning: DeepSeek-V3 (Zen1-Base) model not found locally"
    echo "You may need to download it from Hugging Face first"
fi

if [ ! -d "Qwen/Qwen3-32B" ]; then
    echo "Warning: Qwen3-32B model not found locally"
    echo "You may need to download it from Hugging Face first"
fi

# Update configuration if needed
step "Reviewing configuration"
echo "Using configuration in config.yml"
echo "You may edit this file to customize the merge process"
read -p "Continue with current configuration? [Y/n] " confirm
if [[ $confirm == [nN] ]]; then
    echo "Please edit config.yml and run this script again"
    exit 0
fi

# Run the merge
step "Running model merge"
chmod +x run_merge.sh
./run_merge.sh || error "Merge failed"

# Update model configuration
step "Updating model configuration"
python update_config.py --model-path "$MERGED_MODEL_DIR" || error "Failed to update configuration"

# Convert for inference
step "Converting model for inference"
mkdir -p "$INFERENCE_DIR"
python ../inference/convert.py \
    --hf-ckpt-path "$MERGED_MODEL_DIR" \
    --save-path "$INFERENCE_DIR" \
    --n-experts 256 \
    --model-parallel 16 || error "Failed to convert model"

# Final steps and instructions
step "Model merge completed successfully!"
echo "The merged model is available in: $MERGED_MODEL_DIR"
echo "The inference-ready model is available in: $INFERENCE_DIR"
echo ""
echo "To test the model interactively, run:"
echo "  ./test_merged_model.sh"
echo ""
echo "To clean up temporary files and free up resources:"
echo "  ./utils.sh cleanup"
