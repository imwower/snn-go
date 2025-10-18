package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/nats-io/nats.go"
)

type ringItem struct {
	Topic string          `json:"topic"`
	Data  json.RawMessage `json:"data"`
	T     int64           `json:"time_unix"`
}

func main() {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	// in-memory ring for recent metrics
	ring := make([]ringItem, 0, 200)
	push := func(topic string, b []byte) {
		ring = append(ring, ringItem{Topic: topic, Data: append([]byte(nil), b...), T: time.Now().Unix()})
		if len(ring) > 200 {
			ring = ring[len(ring)-200:]
		}
	}

	// SSE endpoint
	http.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flusher", http.StatusInternalServerError)
			return
		}

		nc, err := nats.Connect(cfg.NATS.URL)
		if err != nil {
			http.Error(w, "nats connect failed", http.StatusInternalServerError)
			return
		}
		defer nc.Drain()

		ch := make(chan *nats.Msg, 256)
		subjects := []string{
			"snn.metrics.batch",
			"snn.metrics.epoch",
			"snn.ui.log.training",
		}
		for _, s := range subjects {
			if _, err := nc.ChanSubscribe(s, ch); err != nil {
				log.Printf("subscribe %s: %v", s, err)
			}
		}
		notify := r.Context().Done()

		for {
			select {
			case msg := <-ch:
				var evType string
				switch msg.Subject {
				case "snn.metrics.batch":
					evType = "metrics_batch"
				case "snn.metrics.epoch":
					evType = "metrics_epoch"
				case "snn.ui.log.training":
					evType = "log"
				default:
					evType = "other"
				}
				// push to ring
				push(evType, msg.Data)
				// write SSE
				w.Write([]byte("event: " + evType + "\n"))
				w.Write([]byte("data: " + string(msg.Data) + "\n\n"))
				flusher.Flush()
			case <-notify:
				return
			case <-time.After(15 * time.Second):
				// keep-alive
				w.Write([]byte(": ping\n\n"))
				flusher.Flush()
			}
		}
	})

	http.HandleFunc("/api/metrics/recent", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ring)
	})

	// serve Vue dist
	dist := "ui-vue/dist"
	index := dist + "/index.html"
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if _, err := os.Stat(index); err == nil {
			http.FileServer(http.Dir(dist)).ServeHTTP(w, r)
			return
		}
		// fallback: hint to build
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write([]byte("Build the UI first: cd ui-vue && npm i && npm run build\n"))
	})

	log.Printf("UI/API listening on %s", cfg.UI.Addr)
	log.Fatal(http.ListenAndServe(cfg.UI.Addr, nil))
}
