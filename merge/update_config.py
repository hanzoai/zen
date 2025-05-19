#!/usr/bin/env python
# Script to update configuration for merged model

import argparse
import json
import os

def update_config(model_path, add_qwen_tokens=False):
    """
    Update the model configuration to support merged model features.
    
    Args:
        model_path: Path to the merged model
        add_qwen_tokens: Whether to add Qwen3 special tokens to the configuration
    """
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Updating model configuration...")
        
        # Ensure the MoE configuration is correctly set
        if "moe_config" not in config:
            print("Adding MoE configuration")
            config["moe_config"] = {
                "n_experts": 256,
                "experts_per_token": 8,
                "use_balanced_experts": True
            }
        
        # Add Qwen3 special tokens if requested
        if add_qwen_tokens:
            print("Adding Qwen3 special tokens")
            if "tokenizer_config" not in config:
                config["tokenizer_config"] = {}
            
            if "special_tokens" not in config["tokenizer_config"]:
                config["tokenizer_config"]["special_tokens"] = {}
            
            # Add Qwen3 special tokens for thinking mode
            config["tokenizer_config"]["special_tokens"].update({
                "/think": {
                    "id": config.get("vocab_size", 32000) + 1,
                    "special": True
                },
                "/no_think": {
                    "id": config.get("vocab_size", 32000) + 2,
                    "special": True
                }
            })
            
            # Update tokenizer configuration to handle special tokens
            config["tokenizer_config"]["add_special_tokens"] = True
            config["tokenizer_config"]["add_bos_token"] = True
        
        # Save updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration updated successfully at {config_path}")
        return True
    
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update merged model configuration")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the merged model")
    parser.add_argument("--add-qwen-tokens", action="store_true", help="Add Qwen3 special tokens")
    
    args = parser.parse_args()
    update_config(args.model_path, args.add_qwen_tokens)
