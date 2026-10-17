// Hanzo Zen Go Quickstart.
// Connects to /v1/chat/completions using standard net/http.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
}

type Choice struct {
	Message Message `json:"message"`
}

type ChatResponse struct {
	Choices []Choice `json:"choices"`
}

func main() {
	apiHost := os.Getenv("HANZO_API_HOST")
	if apiHost == "" {
		apiHost = "https://api.hanzo.ai"
	}
	apiKey := os.Getenv("HANZO_API_KEY")

	fmt.Printf("Connecting to Zen at %s...\n", apiHost)

	reqBody := ChatRequest{
		Model: "zen",
		Messages: []Message{
			{Role: "system", Content: "You are Hanzo Zen."},
			{Role: "user", Content: "Give one tip for low-latency inference."},
		},
		MaxTokens:   128,
		Temperature: 0.1,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		panic(err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimRight(apiHost, "/"))
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		panic(err)
	}

	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Error: HTTP %d: %s\n", resp.StatusCode, string(bodyBytes))
		os.Exit(1)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		panic(err)
	}

	if len(chatResp.Choices) > 0 {
		fmt.Printf("\nResponse:\n%s\n", chatResp.Choices[0].Message.Content)
	}
}
