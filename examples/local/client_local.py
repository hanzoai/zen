#!/usr/bin/env python3
"""Query a locally hosted model with OpenAI-compatible API.

Zero cloud dependencies, zero external network requests.
Works with local servers (llama-server, vLLM, Ollama) or on-prem nodes like evo.local:8730.
"""

import sys
from openai import OpenAI

# Point to localhost or on-premise hardware node (e.g. http://evo.local:8730/v1)
LOCAL_BASE_URL = "http://localhost:8080/v1"

# Any non-empty API key satisfies standard OpenAI client libraries for local servers
client = OpenAI(
    base_url=LOCAL_BASE_URL,
    api_key="local-no-key-needed",
)

def main():
    print(f"Connecting to local inference engine at {LOCAL_BASE_URL}...")
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a concise engineering assistant running locally."},
                {"role": "user", "content": "Summarize the benefit of running on-premise AI models."},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        print("\n=== Local Response ===")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error querying local server: {e}", file=sys.stderr)
        print(f"Ensure local server is running at {LOCAL_BASE_URL} or update URL to evo.local:8730/v1", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
