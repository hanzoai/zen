#!/bin/bash
# Setup script for installing Mergekit with MoE support
set -e  # Exit on any error

echo "=== Setting up Mergekit for DeepSeek-V3 and Qwen3 merge ==="

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Checking memory..."
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    echo "Total GPU memory: ${GPU_MEM}MB"
    
    if [ "$GPU_MEM" -lt 90000 ]; then
        echo "WARNING: This merge requires at least 90GB of GPU memory. You may encounter OOM errors."
    fi
else
    echo "WARNING: No GPU detected. This merge requires multiple high-end GPUs."
fi

# Create a virtual environment
echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Clone mergekit repository with MoE support
echo "Cloning Mergekit repository..."
if [ -d "mergekit" ]; then
    echo "Mergekit directory already exists, updating..."
    cd mergekit
    git fetch
    git switch mixtral  # For MoE functionality
    git pull
    cd ..
else
    git clone https://github.com/arcee-ai/mergekit
    cd mergekit
    git switch mixtral  # For MoE functionality
    git pull
    cd ..
fi

# Install mergekit and dependencies
echo "Installing Mergekit and dependencies..."
cd mergekit
pip install -e .
cd ..
pip install scipy torch>=2.0.0 bitsandbytes>=0.39.0 accelerate>=0.20.0

# Install additional libraries for optimized performance
echo "Installing optimization libraries..."
pip install ninja triton 

# Install libraries for visualization and analysis
echo "Installing analysis libraries..."
pip install pandas matplotlib evaluate

# Check if model weights are available
echo "Checking for model availability..."
if huggingface-cli whoami &> /dev/null; then
    echo "Hugging Face CLI authenticated. You can proceed with the merge."
else
    echo "NOTE: You may need to authenticate with Hugging Face to download the models:"
    echo "Run: huggingface-cli login"
fi

echo ""
echo "Mergekit installation complete!"
echo "Next steps:"
echo "1. Run './run_merge.sh' to start the merging process"
echo "2. After merging, run './test_merged_model.sh' to test the merged model"
echo ""
echo "NOTE: This process requires significant computational resources and may take several hours to days depending on your hardware."

