package anthropic

import (
	"encoding/json"
	"log/slog"
)

const (
	// Upstream model context window size (JoyCode's ModelArts limit).
	// This MUST stay >= the Ctx we advertise to clients in pkg/openai/types.go
	// (currently 200000). Clients (e.g. Claude Code) decide when to auto-compact
	// based on that advertised window, so our defensive truncation threshold must
	// sit ABOVE the client's compact point — otherwise the proxy destructively
	// truncates before the client ever gets a chance to compact cleanly.
	contextWindowSize = 200000
	// Safety margin: only kick in as a last-resort fuse once estimated tokens
	// exceed this ratio of contextWindowSize. Kept high (0.95) so the client's
	// own compaction (which triggers earlier) is always the primary mechanism;
	// this truncation is the backstop for oversized single messages / tool
	// results the client cannot compact away.
	preemptiveThresholdRatio = 0.95
	// Rough approximation: 1 token ≈ 3.5 bytes for mixed Chinese/English code content
	bytesPerToken = 3.5
	// Safety bound only; normal truncation finishes in far fewer rounds.
	maxTruncationRounds = 32
	// Large tool output can otherwise make the retained tail impossible to send.
	maxRetainedToolResultBytes = 32 * 1024
)

// estimateTokens gives a rough token count estimate for the request messages.
// Uses byte-length / bytesPerToken as approximation. Overestimates slightly which is safe.
func estimateTokens(req *MessageRequest) int {
	totalBytes := 0
	if req.System != nil {
		totalBytes += len(req.System)
	}
	for _, m := range req.Messages {
		totalBytes += len(m.Content)
	}
	if totalBytes == 0 {
		return 0
	}
	return int(float64(totalBytes) / bytesPerToken)
}

// PreemptiveTruncate checks if the request likely exceeds the model's context limit
// and proactively truncates before sending. Returns the number of truncation rounds performed.
// Returns -1 if truncation failed to bring estimated tokens below threshold.
func PreemptiveTruncate(req *MessageRequest) int {
	window, ratio := float64(contextWindowSize), float64(preemptiveThresholdRatio)
	threshold := int(window * ratio)
	rounds := 0

	for rounds < maxTruncationRounds {
		estimated := EstimateRequestTokens(req)
		if estimated <= threshold {
			if rounds > 0 {
				slog.Info("preemptive truncation complete", "rounds", rounds, "estimated_tokens", estimated, "threshold", threshold)
			}
			return rounds
		}

		changed := truncateOversizedToolResults(req)
		if !changed {
			changed = truncateMessages(req)
		}
		if !changed {
			logTruncationBreakdown(req, estimated, threshold, rounds)
			return -1
		}

		after := EstimateRequestTokens(req)
		rounds++
		slog.Warn("preemptive truncation round", "round", rounds, "estimated_tokens_before", estimated, "estimated_tokens_after", after, "threshold", threshold)
		if after >= estimated {
			logTruncationBreakdown(req, after, threshold, rounds)
			return -1
		}
	}

	logTruncationBreakdown(req, EstimateRequestTokens(req), threshold, rounds)
	return -1
}

func logTruncationBreakdown(req *MessageRequest, estimated, threshold, rounds int) {
	maxMessageBytes, maxMessageIndex := 0, -1
	for i, message := range req.Messages {
		if len(message.Content) > maxMessageBytes {
			maxMessageBytes = len(message.Content)
			maxMessageIndex = i
		}
	}
	toolsBytes := 0
	if data, err := json.Marshal(req.Tools); err == nil {
		toolsBytes = len(data)
	}
	slog.Warn("preemptive truncation cannot reach threshold",
		"estimated_tokens", estimated,
		"threshold", threshold,
		"rounds", rounds,
		"messages", len(req.Messages),
		"system_bytes", len(req.System),
		"tools_bytes", toolsBytes,
		"max_message_index", maxMessageIndex,
		"max_message_bytes", maxMessageBytes,
	)
}

// findToolPairBoundary adjusts cutEnd so the retained suffix does not start with
// a tool_result whose matching tool_use has been removed.
func findToolPairBoundary(messages []MessageParam, cutEnd int) int {
	if cutEnd <= 0 || cutEnd >= len(messages) {
		return cutEnd
	}
	prevRole := messages[cutEnd-1].Role
	curRole := messages[cutEnd].Role

	if prevRole == "assistant" && curRole == "user" {
		var blocks []contentBlock
		if json.Unmarshal(messages[cutEnd-1].Content, &blocks) == nil {
			for _, b := range blocks {
				if b.Type == "tool_use" {
					if cutEnd > 1 {
						return cutEnd - 1
					}
					if cutEnd+1 < len(messages) {
						return cutEnd + 1
					}
				}
			}
		}
	}

	if curRole == "user" {
		var blocks []contentBlock
		if json.Unmarshal(messages[cutEnd].Content, &blocks) == nil {
			for _, b := range blocks {
				if b.Type == "tool_result" && cutEnd+1 < len(messages) {
					return cutEnd + 1
				}
			}
		}
	}
	return cutEnd
}

// truncateMessages removes the oldest 40% of the conversation. The first
// message is intentionally not pinned: a very large initial prompt was the
// reason repeated truncation previously stalled above the threshold.
func truncateMessages(req *MessageRequest) bool {
	n := len(req.Messages)
	if n <= 1 {
		return false
	}

	cutEnd := int(float64(n) * 0.4)
	if cutEnd < 1 {
		cutEnd = 1
	}
	cutEnd = findToolPairBoundary(req.Messages, cutEnd)
	if cutEnd <= 0 || cutEnd >= n {
		return false
	}

	notice := "[System: Earlier conversation messages have been auto-truncated to fit within the model's context window. Some earlier context is now missing. Continue with the remaining conversation.]"
	noticeBytes, _ := json.Marshal(notice)
	truncated := make([]MessageParam, 0, n-cutEnd+1)
	truncated = append(truncated, MessageParam{Role: "user", Content: json.RawMessage(noticeBytes)})
	truncated = append(truncated, req.Messages[cutEnd:]...)

	slog.Warn("auto-truncated messages for context limit",
		"original_count", n,
		"truncated_count", len(truncated),
		"removed", cutEnd,
		"kept_last", n-cutEnd,
	)
	req.Messages = truncated
	return true
}

// truncateOversizedToolResults replaces large retained tool outputs with a
// marker while preserving the tool_result block and its tool_use_id.
func truncateOversizedToolResults(req *MessageRequest) bool {
	for messageIndex := range req.Messages {
		var blocks []contentBlock
		if json.Unmarshal(req.Messages[messageIndex].Content, &blocks) != nil {
			continue
		}
		changed := false
		for blockIndex := range blocks {
			if blocks[blockIndex].Type != "tool_result" || len(blocks[blockIndex].Content) <= maxRetainedToolResultBytes {
				continue
			}
			blocks[blockIndex].Content = json.RawMessage(`"[Tool result auto-truncated by proxy because it exceeded the context window.]"`)
			changed = true
		}
		if changed {
			content, err := json.Marshal(blocks)
			if err != nil {
				continue
			}
			req.Messages[messageIndex].Content = content
			return true
		}
	}
	return false
}
