#!/bin/bash
# Comprehensive evaluation script for the ZENITH merged model
set -e  # Exit on error

# Configuration
MODEL_PATH="./merged_model_demo"  # Path to the converted model
RESULT_DIR="./eval_results"       # Directory to store evaluation results
TEST_CASES=(
  "code"              # Code generation tasks
  "math"              # Mathematical reasoning
  "reasoning"         # General reasoning
  "thinking"          # Tests for Qwen3's thinking mode
)

# Create results directory
mkdir -p $RESULT_DIR

function print_header() {
  echo "=========================================================="
  echo "  $1"
  echo "=========================================================="
}

function run_evaluation() {
  local test_type=$1
  local prompt_file="eval_prompts_${test_type}.txt"
  local result_file="${RESULT_DIR}/${test_type}_results.json"
  
  # Create prompt file
  case $test_type in
    "code")
      cat > $prompt_file << EOL
Write a Python function to find the longest common subsequence of two strings.

Create a JavaScript function that implements a debounce utility.

Implement a binary search tree in C++ with insert, delete, and search operations.

Create a React component that fetches data from an API and displays it in a paginated table.
EOL
      ;;
    "math")
      cat > $prompt_file << EOL
/think Solve the following problem step by step: If a train travels at 60mph for 2 hours and then at 80mph for 3 hours, what is the average speed for the entire journey?

/think A circle has radius r. A square is inscribed in the circle. What is the ratio of the area of the square to the area of the circle?

/think The sum of three consecutive integers is 51. What are the three integers?

/think In a right triangle, one leg is 6 cm and the hypotenuse is 10 cm. What is the length of the other leg?
EOL
      ;;
    "reasoning")
      cat > $prompt_file << EOL
If all A are B, and all B are C, what can we definitely conclude?

In a race, Tom finished ahead of Jack, and Bill finished behind Maria. Jack finished ahead of Maria. Who finished last?

A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?
EOL
      ;;
    "thinking")
      cat > $prompt_file << EOL
/think Analyze the following argument: "If it's raining, the streets are wet. The streets are wet. Therefore, it's raining." Is this a valid argument? Explain why or why not.

/no_think What is the capital of France?

/think What would happen if everyone always told the truth? Consider multiple aspects such as social interactions, business, politics, and personal relationships.

/no_think List the first five prime numbers.
EOL
      ;;
  esac
  
  print_header "Running evaluation: $test_type"
  echo "Using prompts from: $prompt_file"
  echo "Saving results to: $result_file"
  
  # Run the evaluation
  torchrun --nproc-per-node=4 ../inference/generate.py \
    --ckpt-path $MODEL_PATH \
    --config ../inference/configs/config_671B.json \
    --prompt-file $prompt_file \
    --temperature 0.7 \
    --max-new-tokens 800 \
    --output-file $result_file
    
  echo "Evaluation for $test_type completed"
}

# Main evaluation pipeline
print_header "ZENITH Model Evaluation"
echo "Model path: $MODEL_PATH"
echo "Results directory: $RESULT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path does not exist: $MODEL_PATH"
  echo "Please run test_merged_model.sh first to convert and prepare the model."
  exit 1
fi

# Run evaluations for all test cases
for test_case in "${TEST_CASES[@]}"; do
  run_evaluation $test_case
done

# Generate a summary report
print_header "Evaluation Summary"
echo "$(date): Evaluation completed for ZENITH model" > "${RESULT_DIR}/summary.txt"
echo "Model: $MODEL_PATH" >> "${RESULT_DIR}/summary.txt"
echo "" >> "${RESULT_DIR}/summary.txt"
echo "Test Cases:" >> "${RESULT_DIR}/summary.txt"
for test_case in "${TEST_CASES[@]}"; do
  echo "- $test_case: Results in ${test_case}_results.json" >> "${RESULT_DIR}/summary.txt"
done

echo "Evaluation completed. Summary saved to ${RESULT_DIR}/summary.txt"
