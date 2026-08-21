package main

import (
	"fmt"
	"os"

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
	models, err := c.ListModels()
	if err != nil {
		fmt.Println("ListModels error:", err)
		return
	}
	for _, m := range models {
		fmt.Println(m.ModelID, "|", m.Label)
	}
}