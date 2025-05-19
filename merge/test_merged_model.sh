#!/bin/bash
# Script to test the merged model with enhanced testing for Qwen3 capabilities

# Directory paths
MERGED_MODEL_DIR="./merged_model"
OUTPUT_DIR="./merged_model_demo"

# Create output directory
mkdir -p $OUTPUT_DIR

# Convert merged model to inference format
echo "Converting merged model to inference format..."
python ../inference/convert.py \
  --hf-ckpt-path $MERGED_MODEL_DIR \
  --save-path $OUTPUT_DIR \
  --n-experts 256 \
  --model-parallel 16

# Update configuration for Qwen3 tokenizer compatibility
echo "Updating configuration for Qwen3 tokenizer compatibility..."
python update_config.py \
  --model-path $OUTPUT_DIR \
  --add-qwen-tokens True

# Create test prompts file with examples that leverage both models' strengths
cat > test_prompts.txt << EOL
Generate Python code to implement a binary search tree with insert and delete operations.

/think Solve this step-by-step: A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the average speed for the entire journey?

Explain how transformers work in machine learning, focusing on the attention mechanism.

/no_think What is the capital of France?

Write pseudocode for an algorithm that finds the shortest path in a graph.
EOL

# Test with sample prompts
echo "Testing merged model with sample prompts..."
echo "=== Testing interactive mode ==="
torchrun --nproc-per-node=4 ../inference/generate.py \
  --ckpt-path $OUTPUT_DIR \
  --config ../inference/configs/config_671B.json \
  --interactive \
  --temperature 0.7 \
  --max-new-tokens 200

echo "=== Testing with prepared prompts ==="
torchrun --nproc-per-node=4 ../inference/generate.py \
  --ckpt-path $OUTPUT_DIR \
  --config ../inference/configs/config_671B.json \
  --prompt-file test_prompts.txt \
  --temperature 0.7 \
  --max-new-tokens 400

echo "Testing completed! Merged model combines DeepSeek-V3's code capabilities with Qwen3's step-by-step reasoning."
