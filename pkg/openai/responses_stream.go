package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// chatStreamChunk is the subset of an upstream Chat Completions SSE chunk we
// need to reconstruct Responses events.
type chatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// streamToolCall accumulates a function call across delta chunks.
type streamToolCall struct {
	itemID      string
	callID      string
	name        string
	arguments   strings.Builder
	outputIndex int
	added       bool
}

// handleStreamResponses converts the upstream Chat Completions SSE stream into
// an OpenAI Responses event stream that Codex CLI understands.
func (s *Server) handleStreamResponses(w http.ResponseWriter, r *http.Request, client *joycode.Client, jcBody map[string]interface{}, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		slog.Error("streaming not supported by response writer")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "close")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(200)

	respID := newResponseID()
	created := time.Now().Unix()
	seq := 0
	emit := func(eventType string, payload map[string]interface{}) {
		payload["type"] = eventType
		payload["sequence_number"] = seq
		seq++
		b, _ := json.Marshal(payload)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, b)
		flusher.Flush()
	}

	baseResponse := func(status string) map[string]interface{} {
		return map[string]interface{}{
			"id":         respID,
			"object":     "response",
			"created_at": created,
			"status":     status,
			"model":      model,
			"output":     []interface{}{},
		}
	}

	// response.created
	emit("response.created", map[string]interface{}{"response": baseResponse("in_progress")})
	emit("response.in_progress", map[string]interface{}{"response": baseResponse("in_progress")})

	// Heartbeat while the upstream buffers the full response before first byte.
	stopHeartbeat := make(chan struct{})
	heartbeatDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		defer close(heartbeatDone)
		for {
			select {
			case <-stopHeartbeat:
				return
			case <-ticker.C:
				if _, err := w.Write([]byte(": keepalive\n\n")); err != nil {
					return
				}
				flusher.Flush()
			}
		}
	}()

	streamStart := time.Now()
	resp, err := client.PostStream("/api/saas/openai/v1/chat/completions", jcBody)
	if err != nil {
		close(stopHeartbeat)
		<-heartbeatDone
		slog.Error("responses stream upstream error", "model", model, "error", err)
		msg := err.Error()
		if isTimeoutError(msg) {
			msg = "上游服务响应超时，请稍后重试。原始错误: " + msg
		}
		failed := baseResponse("failed")
		failed["error"] = map[string]interface{}{"message": msg}
		emit("response.failed", map[string]interface{}{"response": failed})
		return
	}
	defer resp.Body.Close()
	close(stopHeartbeat)
	<-heartbeatDone
	slog.Info("responses stream: connected to upstream", "model", model, "ttfb_ms", time.Since(streamStart).Milliseconds())

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	outputIndex := 0
	// Text message state.
	textItemID := ""
	textStarted := false
	var textBuf strings.Builder
	// Tool call state, keyed by upstream tool_call index.
	toolCalls := map[int]*streamToolCall{}
	var inTk, outTk int
	finishReason := ""

	// completedItems collects finished output items (text + tool calls) in
	// emission order for the final response snapshot.
	var completedItems []interface{}

	closeText := func() {
		if !textStarted {
			return
		}
		finalText := textBuf.String()
		emit("response.output_text.done", map[string]interface{}{
			"item_id":      textItemID,
			"output_index": outputIndex,
			"content_index": 0,
			"text":         finalText,
		})
		emit("response.content_part.done", map[string]interface{}{
			"item_id":       textItemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"part":          map[string]interface{}{"type": "output_text", "text": finalText},
		})
		emit("response.output_item.done", map[string]interface{}{
			"output_index": outputIndex,
			"item": map[string]interface{}{
				"id":     textItemID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
				"content": []interface{}{
					map[string]interface{}{"type": "output_text", "text": finalText},
				},
			},
		})
		completedItems = append(completedItems, map[string]interface{}{
			"id":     textItemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []interface{}{
				map[string]interface{}{"type": "output_text", "text": finalText},
			},
		})
		textStarted = false
		outputIndex++
	}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if strings.TrimSpace(data) == "[DONE]" {
			break
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Usage != nil {
			inTk = chunk.Usage.PromptTokens
			outTk = chunk.Usage.CompletionTokens
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			finishReason = choice.FinishReason
		}

		// Text delta.
		if choice.Delta.Content != "" {
			// A text message and a tool call cannot share the same output slot.
			if !textStarted {
				textItemID = fmt.Sprintf("msg_%d", time.Now().UnixNano())
				textStarted = true
				textBuf.Reset()
				emit("response.output_item.added", map[string]interface{}{
					"output_index": outputIndex,
					"item": map[string]interface{}{
						"id":      textItemID,
						"type":    "message",
						"status":  "in_progress",
						"role":    "assistant",
						"content": []interface{}{},
					},
				})
				emit("response.content_part.added", map[string]interface{}{
					"item_id":       textItemID,
					"output_index":  outputIndex,
					"content_index": 0,
					"part":          map[string]interface{}{"type": "output_text", "text": ""},
				})
			}
			textBuf.WriteString(choice.Delta.Content)
			emit("response.output_text.delta", map[string]interface{}{
				"item_id":       textItemID,
				"output_index":  outputIndex,
				"content_index": 0,
				"delta":         choice.Delta.Content,
			})
		}

		// Tool call deltas.
		for _, tc := range choice.Delta.ToolCalls {
			// Once a tool call starts, close any open text item first.
			closeText()
			st, exists := toolCalls[tc.Index]
			if !exists {
				st = &streamToolCall{
					itemID:      fmt.Sprintf("fc_%d_%d", time.Now().UnixNano(), tc.Index),
					outputIndex: outputIndex,
				}
				toolCalls[tc.Index] = st
				outputIndex++
			}
			if tc.ID != "" {
				st.callID = tc.ID
			}
			if tc.Function.Name != "" {
				st.name = tc.Function.Name
			}
			if !st.added && st.name != "" {
				st.added = true
				callID := st.callID
				if callID == "" {
					callID = st.itemID
				}
				emit("response.output_item.added", map[string]interface{}{
					"output_index": st.outputIndex,
					"item": map[string]interface{}{
						"id":        st.itemID,
						"type":      "function_call",
						"status":    "in_progress",
						"name":      st.name,
						"call_id":   callID,
						"arguments": "",
					},
				})
			}
			if tc.Function.Arguments != "" {
				st.arguments.WriteString(tc.Function.Arguments)
				if st.added {
					emit("response.function_call_arguments.delta", map[string]interface{}{
						"item_id":      st.itemID,
						"output_index": st.outputIndex,
						"delta":        tc.Function.Arguments,
					})
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		slog.Error("responses stream read error", "model", model, "error", err)
	}

	// Finalize any open text item.
	closeText()

	// Finalize tool calls (emit args.done + output_item.done), ordered by the
	// upstream tool_call index so output_index stays stable.
	indices := make([]int, 0, len(toolCalls))
	for idx := range toolCalls {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		st := toolCalls[idx]
		callID := st.callID
		if callID == "" {
			callID = st.itemID
		}
		if st.added {
			emit("response.function_call_arguments.done", map[string]interface{}{
				"item_id":      st.itemID,
				"output_index": st.outputIndex,
				"arguments":    st.arguments.String(),
			})
			emit("response.output_item.done", map[string]interface{}{
				"output_index": st.outputIndex,
				"item": map[string]interface{}{
					"id":        st.itemID,
					"type":      "function_call",
					"status":    "completed",
					"name":      st.name,
					"call_id":   callID,
					"arguments": st.arguments.String(),
				},
			})
		}
		completedItems = append(completedItems, map[string]interface{}{
			"id":        st.itemID,
			"type":      "function_call",
			"status":    "completed",
			"name":      st.name,
			"call_id":   callID,
			"arguments": st.arguments.String(),
		})
	}

	// response.completed with usage and final output snapshot.
	final := baseResponse("completed")
	final["output"] = completedItems
	if inTk > 0 || outTk > 0 {
		final["usage"] = map[string]interface{}{
			"input_tokens":  inTk,
			"output_tokens": outTk,
			"total_tokens":  inTk + outTk,
		}
		store.SetTokenUsage(r, inTk, outTk)
	}
	slog.Info("responses stream: completed",
		"model", model,
		"finish_reason", finishReason,
		"tool_calls", len(toolCalls),
		"text_len", textBuf.Len(),
		"in_tokens", inTk,
		"out_tokens", outTk,
	)
	emit("response.completed", map[string]interface{}{"response": final})
}