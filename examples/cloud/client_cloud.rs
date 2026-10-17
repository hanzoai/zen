//! Hanzo Zen Rust SDK Quickstart.
//!
//! Submits an OpenAI-compatible chat completion request to POST /v1/chat/completions.

use serde_json::json;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_host = env::var("HANZO_API_HOST").unwrap_or_else(|_| "https://api.hanzo.ai".into());
    let api_key = env::var("HANZO_API_KEY").unwrap_or_default();

    println!("Connecting to Zen at {api_host}...");

    let payload = json!({
        "model": "zen",
        "messages": [
            {"role": "system", "content": "You are Hanzo Zen, an expert AI engineer."},
            {"role": "user", "content": "What is the key advantage of speculative decoding in 1 sentence?"}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    });

    let client = reqwest::blocking::Client::new();
    let mut req = client.post(format!("{}/v1/chat/completions", api_host.trim_end_matches('/')))
        .header("Content-Type", "application/json")
        .json(&payload);

    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    let resp = req.send()?;
    let status = resp.status();
    let body: serde_json::Value = resp.json()?;

    println!("HTTP Status: {status}");
    if let Some(content) = body["choices"][0]["message"]["content"].as_str() {
        println!("Zen Response:\n{}", content.trim());
    } else {
        println!("Full Response:\n{}", serde_json::to_string_pretty(&body)?);
    }

    Ok(())
}
