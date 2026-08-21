package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func main() {
	st, err := store.Open(os.Getenv("HOME") + "/.joycode-proxy/proxy.db")
	if err != nil {
		fmt.Println("open store:", err)
		return
	}
	defer st.Close()
	acct, err := st.GetDefaultAccount()
	if err != nil {
		fmt.Println("query account:", err)
		return
	}
	c := joycode.NewClient(acct.PtKey, acct.UserID)

	for _, arg := range os.Args[1:] {
		if strings.HasPrefix(arg, "anthropic:") {
			probeAnthropic(c, strings.TrimPrefix(arg, "anthropic:"))
			continue
		}
		name := joycode.UpstreamModelID(arg)
		if strings.HasPrefix(arg, "raw:") {
			name = strings.TrimPrefix(arg, "raw:")
		}
		body := map[string]interface{}{
			"model":      name,
			"messages":   []map[string]interface{}{{"role": "user", "content": os.Getenv("PROBE_PROMPT")}},
			"max_tokens": 2000,
		}
		if os.Getenv("PROBE_SKIP_NONSTREAM") != "" {
			body["stream"] = true
			_, err := c.Post("/api/saas/openai/v1/chat/completions", body)
			if err != nil {
				fmt.Printf("%s => ERROR: %v\n", name, err)
				continue
			}
			fmt.Printf("%s => OK (non-stream)\n", name)
		} else {
			body["stream"] = true
		}
		sresp, serr := c.PostStream("/api/saas/openai/v1/chat/completions", body)
		if serr != nil {
			fmt.Printf("%s => STREAM ERROR: %v\n", name, serr)
			continue
		}
		// Timing probe: report when each of the first chunks arrives to see
		// whether the upstream truly streams incrementally or buffers.
		start := time.Now()
		buf := make([]byte, 96*1024)
		chunks := 0
		var firstT, lastT time.Duration
		for {
			n, rerr := sresp.Body.Read(buf)
			if n > 0 {
				if chunks == 0 {
					firstT = time.Since(start)
				}
				chunks++
				lastT = time.Since(start)
				if chunks <= 3 || chunks%200 == 0 {
					fmt.Printf("  chunk#%d at %v\n", chunks, time.Since(start).Round(time.Millisecond))
				}
			}
			if rerr != nil {
				break
			}
		}
		sresp.Body.Close()
		fmt.Printf("%s => STREAM done: chunks=%d first=%v last=%v\n", name, chunks, firstT.Round(time.Millisecond), lastT.Round(time.Millisecond))
	}
}

func probeAnthropic(c *joycode.Client, model string) {
	ab := map[string]interface{}{
		"model":      model,
		"max_tokens": 2000,
		"stream":     true,
		"messages":   []map[string]interface{}{{"role": "user", "content": os.Getenv("PROBE_PROMPT")}},
	}
	if os.Getenv("PROBE_THINKING") == "" {
		ab["thinking"] = map[string]string{"type": "disabled"}
	}
	resp, err := c.PostAnthropicStream("/api/saas/anthropic/v1/messages", ab)
	if err != nil {
		fmt.Println("anthropic stream => ERROR:", err)
		return
	}
	defer resp.Body.Close()
	start := time.Now()
	buf := make([]byte, 96*1024)
	chunks := 0
	var firstT, lastT time.Duration
	var full strings.Builder
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			full.Write(buf[:n])
			if chunks == 0 {
				firstT = time.Since(start)
			}
			chunks++
			lastT = time.Since(start)
			if chunks <= 3 || chunks%100 == 0 {
				fmt.Printf("  anthropic chunk#%d at %v\n", chunks, time.Since(start).Round(time.Millisecond))
			}
		}
		if rerr != nil {
			break
		}
	}
	fmt.Printf("anthropic %s => STREAM done: chunks=%d first=%v last=%v\n", model, chunks, firstT.Round(time.Millisecond), lastT.Round(time.Millisecond))
	fmt.Printf("anthropic %s => body:\n%s\n", model, full.String())
}