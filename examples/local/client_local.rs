//! client_local.rs - Pure Rust client querying a local OpenAI-compatible endpoint.
//!
//! Run with:
//!   cargo run --example client_local

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: OutputMessage,
}

#[derive(Deserialize, Debug)]
struct OutputMessage {
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::var("LOCAL_ENDPOINT").unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".into());
    println!("Connecting to local model endpoint at: {}", endpoint);

    let client = reqwest::Client::new();
    let req = ChatRequest {
        model: "local-model".into(),
        messages: vec![
            Message {
                role: "system".into(),
                content: "You are a concise assistant.".into(),
            },
            Message {
                role: "user".into(),
                content: "Explain how pure Rust inference eliminates GIL bottlenecks.".into(),
            },
        ],
        temperature: 0.7,
    };

    let resp: ChatResponse = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .json(&req)
        .send()
        .await?
        .json()
        .await?;

    if let Some(first) = resp.choices.first() {
        println!("\n=== Local Rust Client Response ===");
        println!("{}", first.message.content);
    }

    Ok(())
}
