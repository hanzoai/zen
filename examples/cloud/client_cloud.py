#!/usr/bin/env python3
"""Query Hanzo Cloud frontier and open-weight models using HANZO_API_KEY.

Endpoint: https://api.hanzo.ai/v1
Models: zen5, zen5-mini, zen5-flash, zen5-coder, enso-auto, enso-ultra, etc.
"""

import os
import sys
from openai import OpenAI

api_key = os.environ.get("HANZO_API_KEY")
if not api_key:
    print("Error: HANZO_API_KEY environment variable is required", file=sys.stderr)
    print("Obtain your key from https://platform.hanzo.ai", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    base_url="https://api.hanzo.ai/v1",
    api_key=api_key,
)

def main():
    print("Connecting to Hanzo Cloud (api.hanzo.ai)...")
    
    # 1. Non-streaming call with zen5
    print("\n1. Standard completion (zen5):")
    response = client.chat.completions.create(
        model="zen5",
        messages=[
            {"role": "system", "content": "You are a helpful and concise AI assistant."},
            {"role": "user", "content": "What is the difference between open-weight serving and closed API endpoints?"},
        ],
        temperature=0.7,
    )
    print(response.choices[0].message.content)

    # 2. Streaming completion with zen5-flash
    print("\n2. Streaming completion (zen5-flash):")
    stream = client.chat.completions.create(
        model="zen5-flash",
        messages=[
            {"role": "user", "content": "Write a haiku about distributed GPU inference."},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
