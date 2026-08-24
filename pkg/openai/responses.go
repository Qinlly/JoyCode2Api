package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// ResponsesRequest is the subset of the OpenAI Responses API request that
// Codex CLI actually sends. `input` is an array of items; `instructions`
// maps to a leading system message; `tools` are Responses-style function
// tool definitions (flat, not nested under a "function" key).
type ResponsesRequest struct {
	Model        string          `json:"model"`
	Input        json.RawMessage `json:"input"`
	Instructions string          `json:"instructions,omitempty"`
	Tools        json.RawMessage `json:"tools,omitempty"`
	ToolChoice   json.RawMessage `json:"tool_choice,omitempty"`
	Stream       bool            `json:"stream,omitempty"`
	MaxTokens    int             `json:"max_output_tokens,omitempty"`
	Temperature  *float64        `json:"temperature,omitempty"`
	TopP         *float64        `json:"top_p,omitempty"`
	Reasoning    json.RawMessage `json:"reasoning,omitempty"`
}

// responsesInputItem is a single item in the Responses `input` array.
// Codex uses: {type:"message", role, content:[{type,text}]},
// {type:"function_call", name, arguments, call_id},
// {type:"function_call_output", call_id, output}.
type responsesInputItem struct {
	Type    string          `json:"type"`
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
	// function_call fields
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	CallID    string `json:"call_id"`
	// function_call_output fields
	Output json.RawMessage `json:"output"`
	// additional_tools fields (Codex embeds tool definitions inside `input`
	// as {type:"additional_tools", role:"developer", tools:[...namespaces...]}).
	Tools json.RawMessage `json:"tools"`
}

// responsesContentPart is one entry of an item's content array.
type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL string `json:"image_url"`
}

// handleResponses implements POST /v1/responses (OpenAI Responses API),
// translating to the upstream Chat Completions endpoint and converting the
// response back into Responses events (streaming) or object (non-streaming).
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if !requirePOST(w, r) {
		return
	}
	rawBody, _ := io.ReadAll(r.Body)
	var req ResponsesRequest
	if err := json.Unmarshal(rawBody, &req); err != nil {
		slog.Error("decode responses request", "error", err)
		writeError(w, 400, fmt.Sprintf("请求体解析失败: %s。请检查请求是否完整，或尝试开启新对话减少上下文长度。", err.Error()))
		return
	}

	systemDefault := ""
	if s.store != nil {
		systemDefault = s.store.GetSetting("default_model")
	}
	model := ResolveModel(req.Model, store.GetAccountDefaultModel(r), systemDefault)
	store.SetModel(r, model)

	chatReq, err := responsesToChatRequest(&req, model)
	if err != nil {
		slog.Error("responses -> chat translate", "error", err)
		writeError(w, 400, err.Error())
		return
	}
	jcBody := TranslateRequest(chatReq)

	// DIAG: verify model resolution and tool extraction after this build.
	toolsOut, _ := json.Marshal(jcBody["tools"])
	slog.Info("DIAG responses resolved", "req_model", req.Model, "resolved", model,
		"tools_len", len(chatReq.Tools), "tools", string(toolsOut))

	client := s.getClient(r)

	if req.Stream {
		s.handleStreamResponses(w, r, client, jcBody, model)
	} else {
		s.handleNonStreamResponses(w, r, client, jcBody, model)
	}
}

// responsesToChatRequest converts a Responses request into the internal
// ChatRequest so it can reuse TranslateRequest and the upstream chat path.
func responsesToChatRequest(req *ResponsesRequest, model string) (*ChatRequest, error) {
	var messages []map[string]interface{}

	if strings.TrimSpace(req.Instructions) != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": req.Instructions,
		})
	}

	items, err := parseInputItems(req.Input)
	if err != nil {
		return nil, err
	}
	// Codex (Responses protocol) embeds its tool definitions inside the
	// `input` array as `additional_tools` items rather than the top-level
	// `tools` field. Collect them here so they survive translation.
	var embeddedTools []map[string]interface{}
	for _, it := range items {
		if it.Type == "additional_tools" {
			embeddedTools = append(embeddedTools, extractAdditionalTools(it.Tools)...)
			continue
		}
		msg := inputItemToMessage(it)
		if msg != nil {
			messages = append(messages, msg)
		}
	}

	msgBytes, err := json.Marshal(messages)
	if err != nil {
		return nil, fmt.Errorf("marshal messages: %w", err)
	}

	cr := &ChatRequest{
		Model:       model,
		Messages:    msgBytes,
		Stream:      req.Stream,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}
	// Merge tools from the top-level `tools` field and any embedded in the
	// `input` array (Codex additional_tools). Embedded tools take the same
	// Chat-function shape after conversion.
	var allTools []map[string]interface{}
	if tools := convertResponsesTools(req.Tools); tools != nil {
		var top []map[string]interface{}
		if json.Unmarshal(tools, &top) == nil {
			allTools = append(allTools, top...)
		}
	}
	allTools = append(allTools, embeddedTools...)
	if len(allTools) > 0 {
		cr.Tools = json.RawMessage(mustMarshal(allTools))
	}
	if len(req.ToolChoice) > 0 {
		cr.ToolChoice = req.ToolChoice
	}
	return cr, nil
}

// extractAdditionalTools flattens a Codex `additional_tools` item's tools
// array into Chat-style function tools. The structure is a list of namespaces
// each holding leaf tools:
//
//	[{type:"namespace", name:"functions", tools:[
//	    {type:"custom", name:"exec", description:"..."},
//	    {type:"function", name:"...", parameters:{...}},
//	]}]
//
// `custom` tools accept freeform text (no JSON schema). Upstream Chat
// Completions only understands standard function tools, so we down-convert a
// `custom` tool into a function that takes a single `input` string argument.
func extractAdditionalTools(raw json.RawMessage) []map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	var entries []map[string]interface{}
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil
	}
	var out []map[string]interface{}
	for _, e := range entries {
		typ, _ := e["type"].(string)
		switch typ {
		case "namespace":
			// Recurse into nested tools.
			if nested, ok := e["tools"]; ok {
				nb := mustMarshal(nested)
				out = append(out, extractAdditionalTools(json.RawMessage(nb))...)
			}
		case "custom":
			out = append(out, customToolToFunction(e))
		case "function":
			out = append(out, normalizeFunctionTool(e))
		}
	}
	return out
}

// customToolToFunction converts a Codex freeform `custom` tool into a standard
// Chat function tool with a single freeform string `input` parameter.
func customToolToFunction(t map[string]interface{}) map[string]interface{} {
	fn := map[string]interface{}{}
	if v, ok := t["name"]; ok {
		fn["name"] = v
	}
	if v, ok := t["description"]; ok {
		fn["description"] = v
	}
	fn["parameters"] = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"input": map[string]interface{}{
				"type":        "string",
				"description": "Raw freeform input passed to the tool.",
			},
		},
		"required": []string{"input"},
	}
	return map[string]interface{}{"type": "function", "function": fn}
}

// normalizeFunctionTool reshapes a flat Responses function tool into the
// Chat nested form (mirrors convertResponsesTools for a single entry).
func normalizeFunctionTool(t map[string]interface{}) map[string]interface{} {
	if _, nested := t["function"]; nested {
		return t
	}
	fn := map[string]interface{}{}
	if v, ok := t["name"]; ok {
		fn["name"] = v
	}
	if v, ok := t["description"]; ok {
		fn["description"] = v
	}
	if v, ok := t["parameters"]; ok {
		fn["parameters"] = v
	}
	if v, ok := t["strict"]; ok {
		fn["strict"] = v
	}
	return map[string]interface{}{"type": "function", "function": fn}
}

// parseInputItems accepts either a plain string (shorthand for a single user
// message) or an array of Responses input items.
func parseInputItems(raw json.RawMessage) ([]responsesInputItem, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	trimmed := strings.TrimSpace(string(raw))
	if strings.HasPrefix(trimmed, "\"") {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, fmt.Errorf("parse input string: %w", err)
		}
		return []responsesInputItem{{
			Type: "message",
			Role: "user",
			Content: json.RawMessage(mustMarshal([]responsesContentPart{
				{Type: "input_text", Text: s},
			})),
		}}, nil
	}
	var items []responsesInputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("parse input items: %w", err)
	}
	return items, nil
}

// inputItemToMessage converts one Responses input item into a Chat message.
func inputItemToMessage(it responsesInputItem) map[string]interface{} {
	switch it.Type {
	case "message", "":
		role := it.Role
		if role == "" {
			role = "user"
		}
		// developer role maps to system for chat compatibility.
		if role == "developer" {
			role = "system"
		}
		content := parseContentParts(it.Content, role)
		return map[string]interface{}{"role": role, "content": content}

	case "function_call":
		// Assistant message carrying a tool call.
		return map[string]interface{}{
			"role":    "assistant",
			"content": nil,
			"tool_calls": []map[string]interface{}{{
				"id":   it.CallID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      it.Name,
					"arguments": it.Arguments,
				},
			}},
		}

	case "function_call_output":
		return map[string]interface{}{
			"role":         "tool",
			"tool_call_id": it.CallID,
			"content":      outputToString(it.Output),
		}
	}
	return nil
}

// parseContentParts flattens Responses content parts into either a plain
// string (text-only) or a multimodal content array (with images).
func parseContentParts(raw json.RawMessage, role string) interface{} {
	if len(raw) == 0 {
		return ""
	}
	// content may be a plain string.
	trimmed := strings.TrimSpace(string(raw))
	if strings.HasPrefix(trimmed, "\"") {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
	}
	var parts []responsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	var texts []string
	var multimodal []map[string]interface{}
	hasImage := false
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text", "summary_text":
			texts = append(texts, p.Text)
			multimodal = append(multimodal, map[string]interface{}{
				"type": "text", "text": p.Text,
			})
		case "input_image", "image_url":
			if p.ImageURL != "" {
				hasImage = true
				multimodal = append(multimodal, map[string]interface{}{
					"type":      "image_url",
					"image_url": map[string]interface{}{"url": p.ImageURL},
				})
			}
		}
	}
	if hasImage {
		return multimodal
	}
	return strings.Join(texts, "")
}

// outputToString normalizes a function_call_output value into a string.
func outputToString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	trimmed := strings.TrimSpace(string(raw))
	if strings.HasPrefix(trimmed, "\"") {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
	}
	// Codex may send output as {type:"...", text:"..."} parts or an object.
	var parts []responsesContentPart
	if json.Unmarshal(raw, &parts) == nil && len(parts) > 0 {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		if b.Len() > 0 {
			return b.String()
		}
	}
	return string(raw)
}

// convertResponsesTools converts Responses flat function tools to Chat-style
// nested tools. Responses: {type:"function", name, description, parameters}.
// Chat: {type:"function", function:{name, description, parameters}}.
func convertResponsesTools(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var tools []map[string]interface{}
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil
	}
	var out []map[string]interface{}
	for _, t := range tools {
		typ, _ := t["type"].(string)
		if typ != "function" {
			// Skip non-function tools (web_search, file_search, mcp, etc.).
			continue
		}
		if _, nested := t["function"]; nested {
			out = append(out, t)
			continue
		}
		fn := map[string]interface{}{}
		if v, ok := t["name"]; ok {
			fn["name"] = v
		}
		if v, ok := t["description"]; ok {
			fn["description"] = v
		}
		if v, ok := t["parameters"]; ok {
			fn["parameters"] = v
		}
		if v, ok := t["strict"]; ok {
			fn["strict"] = v
		}
		out = append(out, map[string]interface{}{
			"type": "function", "function": fn,
		})
	}
	if len(out) == 0 {
		return nil
	}
	return json.RawMessage(mustMarshal(out))
}

func mustMarshal(v interface{}) []byte {
	b, _ := json.Marshal(v)
	return b
}

func newResponseID() string {
	return fmt.Sprintf("resp_%d", time.Now().UnixNano())
}