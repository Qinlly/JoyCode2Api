package anthropic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func TestResolveModel(t *testing.T) {
	tests := []struct {
		name           string
		model          string
		accountDefault string
		systemDefault  string
		expected       string
	}{
		{"known joycode model passes through", "GLM-4.7", "", "", "GLM-4.7"},
		{"unknown model falls back to default", "claude-sonnet-4-20250514", "", "", joycode.DefaultModel},
		{"account default overrides for unknown model", "claude-opus-4", "Kimi-K2.6", "GLM-5.1", "Kimi-K2.6"},
		{"system default used when no account default", "unknown-model", "", "GLM-5.1", "GLM-5.1"},
	}
	for _, tt := range tests {
		got := resolveModel(tt.model, tt.accountDefault, tt.systemDefault)
		if got != tt.expected {
			t.Errorf("%s: resolveModel(%q, %q, %q) = %q, want %q",
				tt.name, tt.model, tt.accountDefault, tt.systemDefault, got, tt.expected)
		}
	}
}

func TestTranslateRequest(t *testing.T) {
	temp := 0.7
	req := &MessageRequest{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Temperature: &temp,
	}
	body := TranslateRequest(req, "", "")

	if body["model"] != joycode.DefaultModel {
		t.Errorf("model = %v, want %s", body["model"], joycode.DefaultModel)
	}
	if body["max_tokens"] != 1024 {
		t.Errorf("max_tokens = %v, want 1024", body["max_tokens"])
	}
	if body["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", body["temperature"])
	}
	msgs, ok := body["messages"].([]map[string]interface{})
	if !ok {
		t.Fatalf("messages type = %T, want []map[string]interface{}", body["messages"])
	}
	if len(msgs) != 1 || msgs[0]["role"] != "user" {
		t.Errorf("messages = %v, unexpected", msgs)
	}
}

func TestTranslateRequestWithSystem(t *testing.T) {
	req := &MessageRequest{
		Model:     "claude-sonnet-4",
		MaxTokens: 2048,
		System:    json.RawMessage(`"You are a helpful assistant"`),
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}
	body := TranslateRequest(req, "", "")
	msgs := body["messages"].([]map[string]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(msgs))
	}
	if msgs[0]["role"] != "system" {
		t.Errorf("first message role = %q, want system", msgs[0]["role"])
	}
}

func TestTranslateResponse(t *testing.T) {
	jcResp := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"content": "Hi! How can I help?",
				},
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(10),
			"completion_tokens": float64(5),
		},
	}
	resp := TranslateResponse(jcResp, "claude-sonnet-4-20250514")

	if resp.Type != "message" {
		t.Errorf("Type = %q, want message", resp.Type)
	}
	if resp.Role != "assistant" {
		t.Errorf("Role = %q, want assistant", resp.Role)
	}
	if resp.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q, want claude-sonnet-4-20250514", resp.Model)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %v, want end_turn", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" {
		t.Errorf("Content = %v, unexpected", resp.Content)
	}
	if resp.Content[0].Text != "Hi! How can I help?" {
		t.Errorf("Text = %q, want 'Hi! How can I help?'", resp.Content[0].Text)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", resp.Usage.OutputTokens)
	}
	if !strings.HasPrefix(resp.ID, "msg_") {
		t.Errorf("ID = %q, want msg_ prefix", resp.ID)
	}
}

func TestParseStreamDelta(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`data: {"choices":[{"delta":{"content":"Hello"}}]}`, "Hello"},
		{`data: [DONE]`, ""},
		{`data: {"choices":[]}`, ""},
		{``, ""},
		{`data: {"choices":[{"delta":{"content":" world"}}]}`, " world"},
	}
	for _, tt := range tests {
		got := ParseStreamDelta(tt.input)
		if got != tt.expected {
			t.Errorf("ParseStreamDelta(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestParseContent(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"plain text"`, "plain text"},
		{`[{"type":"text","text":"hello"},{"type":"text","text":"world"}]`, "hello\nworld"},
		{`[{"type":"image","text":"skip me"}]`, ""},
		{``, ""},
	}
	for _, tt := range tests {
		got := parseContent(json.RawMessage(tt.input))
		if got != tt.expected {
			t.Errorf("parseContent(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestHandlerMessagesEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		body       string
		wantStatus int
	}{
		{"OPTIONS preflight", "OPTIONS", "", 200},
		{"GET rejected", "GET", "", 405},
		{"invalid JSON", "POST", `{bad}`, 400},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body strings.Reader
			if tt.body != "" {
				body = *strings.NewReader(tt.body)
			}
			req := httptest.NewRequest(tt.method, "/v1/messages", &body)
			if tt.method == "POST" {
				req.Header.Set("Content-Type", "application/json")
			}
			w := httptest.NewRecorder()
			h := &Handler{Client: nil}
			h.handleMessages(w, req)
			if w.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d; body = %s", w.Code, tt.wantStatus, w.Body.String())
			}
		})
	}
}

func TestCountTokensEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		body       string
		wantStatus int
	}{
		{"OPTIONS preflight", "OPTIONS", "", 200},
		{"GET rejected", "GET", "", 405},
		{"invalid JSON", "POST", `{bad}`, 400},
		{"valid request", "POST", `{"model":"claude-sonnet-4","messages":[{"role":"user","content":"hello"}]}`, 200},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body strings.Reader
			if tt.body != "" {
				body = *strings.NewReader(tt.body)
			}
			req := httptest.NewRequest(tt.method, "/v1/messages/count_tokens", &body)
			if tt.method == "POST" {
				req.Header.Set("Content-Type", "application/json")
			}
			w := httptest.NewRecorder()
			h := &Handler{}
			h.handleCountTokens(w, req)
			if w.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d; body = %s", w.Code, tt.wantStatus, w.Body.String())
			}
			// Only POST responses carry a JSON body; OPTIONS preflight is empty.
			if tt.wantStatus == 200 && tt.method == "POST" {
				var resp struct {
					InputTokens int `json:"input_tokens"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
					t.Fatalf("response is not valid JSON with input_tokens: %v; body = %s", err, w.Body.String())
				}
				if resp.InputTokens <= 0 {
					t.Errorf("input_tokens = %d, want > 0", resp.InputTokens)
				}
			}
		})
	}
}

func TestCountTokensRouteRegistered(t *testing.T) {
	mux := http.NewServeMux()
	h := &Handler{}
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("POST", "/v1/messages/count_tokens",
		strings.NewReader(`{"model":"claude-sonnet-4","messages":[{"role":"user","content":"hello world, this is a test"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v; body = %s", err, w.Body.String())
	}
	if _, ok := resp["input_tokens"]; !ok {
		t.Errorf("missing input_tokens in response: %s", w.Body.String())
	}
}

func TestAnthropicResponseFormat(t *testing.T) {
	jcResp := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"content": "Test response",
				},
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(100),
			"completion_tokens": float64(50),
		},
	}
	resp := TranslateResponse(jcResp, "claude-sonnet-4-20250514")

	// Verify JSON serialization matches Anthropic format
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var parsed map[string]interface{}
	json.Unmarshal(b, &parsed)

	// Check all required Anthropic response fields
	required := []string{"id", "type", "role", "content", "model", "stop_reason", "usage"}
	for _, field := range required {
		if _, ok := parsed[field]; !ok {
			t.Errorf("missing required field: %s in response: %s", field, string(b))
		}
	}

	if parsed["type"] != "message" {
		t.Errorf("type = %v, want message", parsed["type"])
	}
	if parsed["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", parsed["role"])
	}
}
