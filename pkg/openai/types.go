package openai

import "encoding/json"

// ChatRequest is the OpenAI-compatible chat completion request.
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    json.RawMessage `json:"messages"`
	Stream      bool            `json:"stream,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Tools       json.RawMessage `json:"tools,omitempty"`
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	Stop        json.RawMessage `json:"stop,omitempty"`
	Thinking    json.RawMessage `json:"thinking,omitempty"`
}

// ModelCapability describes a model's feature set.
type ModelCapability struct {
	Vision    bool `json:"vision"`
	Reasoning bool `json:"reasoning"`
	MaxTokens int  `json:"max_tokens"`
	Ctx       int  `json:"ctx"`
}

// ModelCapabilities maps model IDs to their capabilities.
var ModelCapabilities = map[string]ModelCapability{
	"JoyAI-Code-1.5":      {MaxTokens: 64000, Ctx: 200000},
	"JoyAI-Code":          {MaxTokens: 64000, Ctx: 200000},
	"MiniMax-M3":          {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"MiniMax-M2.7":        {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"Kimi-K3":             {Vision: true, Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"Kimi-K2.6":           {Vision: true, Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"Kimi-K2.5":           {Vision: true, MaxTokens: 64000, Ctx: 200000},
	"GLM-5.3":             {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"GLM-5.2":             {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"GLM-5.2-jcloud":      {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"GLM-5.1":             {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"GLM-5":               {MaxTokens: 64000, Ctx: 200000},
	"GLM-5-jcloud":        {MaxTokens: 64000, Ctx: 200000},
	"GLM-4.7":             {MaxTokens: 64000, Ctx: 200000},
	"DeepSeek-V4-Pro":     {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"Doubao-Seed-2.0-pro": {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"GPT-5.6 Sol":         {Reasoning: true, MaxTokens: 64000, Ctx: 200000},
	"Claude-Opus-4.8":     {MaxTokens: 64000, Ctx: 200000},
	"Claude-Opus-4.7":     {MaxTokens: 64000, Ctx: 200000},
	"Claude-Sonnet-4.6":   {MaxTokens: 64000, Ctx: 200000},
	"Claude-Opus-4.6":     {MaxTokens: 64000, Ctx: 200000},
	"JoyCode-Base-V3":     {MaxTokens: 4096, Ctx: 4096},
}

// ReasoningModels supports thinking/reasoning control parameters.
var ReasoningModels = map[string]bool{
	"GLM-5.3": true, "GLM-5.2": true, "GLM-5.2-jcloud": true,
	"GLM-5.1": true, "Kimi-K3": true, "Kimi-K2.6": true,
	"MiniMax-M3": true, "MiniMax-M2.7": true,
	"DeepSeek-V4-Pro": true, "Doubao-Seed-2.0-pro": true,
	"GPT-5.6 Sol": true,
}
