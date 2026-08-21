package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// EstimateRequestTokens estimates the total input token count of a request,
// including the system prompt, all messages, tool definitions and tool_choice.
// It reuses the byte-length heuristic from truncate.go (1 token ≈ 3.5 bytes),
// which slightly overestimates — safe for context accounting.
func EstimateRequestTokens(req *MessageRequest) int {
	total := estimateTokens(req)
	if len(req.Tools) > 0 {
		if b, err := json.Marshal(req.Tools); err == nil {
			total += int(float64(len(b)) / bytesPerToken)
		}
	}
	if len(req.ToolChoice) > 0 {
		total += int(float64(len(req.ToolChoice)) / bytesPerToken)
	}
	return total
}

// handleCountTokens implements POST /v1/messages/count_tokens.
//
// Claude Code calls this endpoint to track context usage and decide when to
// auto-compact. Previously the route was not registered, so these requests
// fell through to the static-file fallback which returned 200 + HTML —
// silently breaking auto-compact and letting sessions grow until the upstream
// started dropping connections. We answer with a local byte-based estimate in
// the official response format {"input_tokens": N}.
func (h *Handler) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.WriteHeader(200)
		return
	}
	if r.Method != http.MethodPost {
		writeAnthropicError(w, 405, "method not allowed")
		return
	}
	var req MessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		reqLog(r).Error("decode count_tokens request", "error", err)
		writeAnthropicError(w, 400, fmt.Sprintf("请求体解析失败: %s", err.Error()))
		return
	}
	tokens := EstimateRequestTokens(&req)
	reqLog(r).Info("count_tokens",
		"model", req.Model,
		"messages", len(req.Messages),
		"tools", len(req.Tools),
		"input_tokens", tokens,
	)
	writeAnthropicJSON(w, 200, map[string]int{"input_tokens": tokens})
}