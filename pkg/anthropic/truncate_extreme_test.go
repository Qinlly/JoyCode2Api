package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPreemptiveTruncate_DoesNotPinHugeFirstMessage(t *testing.T) {
	huge := strings.Repeat("x", 700000)
	req := &MessageRequest{
		Messages: []MessageParam{
			{Role: "user", Content: mustRawJSON(t, huge)},
			{Role: "assistant", Content: mustRawJSON(t, "old response")},
			{Role: "user", Content: mustRawJSON(t, "current task")},
		},
	}

	rounds := PreemptiveTruncate(req)
	if rounds < 1 {
		t.Fatalf("expected truncation to succeed, got rounds=%d", rounds)
	}
	threshold := contextWindowSize * 85 / 100
	if got := EstimateRequestTokens(req); got > threshold {
		t.Fatalf("estimated tokens remain above threshold: got %d, threshold %d", got, threshold)
	}
	for _, message := range req.Messages {
		if strings.Contains(string(message.Content), huge[:1000]) {
			t.Fatal("huge first message was unexpectedly retained")
		}
	}
}

func TestPreemptiveTruncate_TruncatesSingleHugeToolResult(t *testing.T) {
	hugeResult := strings.Repeat("tool output ", 70000)
	toolUse := json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"Read","input":{"file_path":"/tmp/data"}}]`)
	toolResult, err := json.Marshal([]map[string]any{{
		"type":        "tool_result",
		"tool_use_id": "tu_1",
		"content":     hugeResult,
	}})
	if err != nil {
		t.Fatal(err)
	}
	req := &MessageRequest{
		Messages: []MessageParam{
			{Role: "assistant", Content: toolUse},
			{Role: "user", Content: toolResult},
		},
	}

	rounds := PreemptiveTruncate(req)
	if rounds < 1 {
		t.Fatalf("expected oversized tool result truncation, got rounds=%d", rounds)
	}
	if strings.Contains(string(req.Messages[len(req.Messages)-1].Content), hugeResult[:1000]) {
		t.Fatal("oversized tool result content was unexpectedly retained")
	}
	if !strings.Contains(string(req.Messages[len(req.Messages)-1].Content), `"tool_use_id":"tu_1"`) {
		t.Fatal("tool_result lost its tool_use_id")
	}
	threshold := contextWindowSize * 85 / 100
	if got := EstimateRequestTokens(req); got > threshold {
		t.Fatalf("estimated tokens remain above threshold: got %d, threshold %d", got, threshold)
	}
}

// A new conversation containing a single large base64 image must NOT be
// rejected as "context too long". The base64 payload is large in bytes but
// costs only a flat per-image token amount.
func TestPreemptiveTruncate_SingleImageIsNotContextOverflow(t *testing.T) {
	bigBase64 := strings.Repeat("A", 1500000) // ~1.5MB base64; ~430k tokens if mis-estimated by bytes
	imageBlock, err := json.Marshal([]map[string]any{
		{
			"type": "image",
			"source": map[string]string{
				"type":       "base64",
				"media_type": "image/png",
				"data":       bigBase64,
			},
		},
		{"type": "text", "text": "这是什么"},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := &MessageRequest{
		Messages: []MessageParam{
			{Role: "user", Content: imageBlock},
		},
	}

	threshold := int(contextWindowSize * preemptiveThresholdRatio)
	if got := EstimateRequestTokens(req); got > threshold {
		t.Fatalf("single image wrongly estimated as overflow: got %d tokens, threshold %d", got, threshold)
	}
	if rounds := PreemptiveTruncate(req); rounds != 0 {
		t.Fatalf("expected no truncation for a single image, got rounds=%d", rounds)
	}
}

func mustRawJSON(t *testing.T, value string) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
