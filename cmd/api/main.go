package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/natsbus"
	"github.com/nats-io/nats.go"
)

func main() {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	bus, err := natsbus.Connect(natsbus.StreamConfig{
		Stream: cfg.NATS.Stream, URL: cfg.NATS.URL, DupeWindowSec: cfg.NATS.DupeWindowSec,
	})
	if err != nil {
		log.Fatalf("nats: %v", err)
	}
	defer bus.Close()

	// 简单内存缓存最近指标用于 /api/metrics/recent
	type item struct {
		Topic string
		Data  json.RawMessage
		T     time.Time
	}
	var ring []item

	http.HandleFunc("/api/metrics/recent", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		_ = enc.Encode(ring)
	})

	// SSE：把 NATS 的 metrics/log 事件转成浏览器推送
	http.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flush", 500)
			return
		}

		// 订阅 JetStream（拉式 -> 简化成常规 sub）
		nc, _ := nats.Connect(cfg.NATS.URL)
		js, _ := nc.JetStream()
		subj := []string{cfg.NATS.Subjects.MetricsBatch, cfg.NATS.Subjects.MetricsEpoch, cfg.NATS.Subjects.UILog}
		subs := make([]*nats.Subscription, 0, len(subj))
		for _, s := range subj {
			sub, _ := js.SubscribeSync(s)
			subs = append(subs, sub)
		}
		defer nc.Drain()

		for {
			msg, err := subs[0].NextMsg(500 * time.Millisecond) // 轮询其中一个，简化示例
			if err == nil {
				// 按 type 包装为 {type: "...", data: ...}
				var evType string
				switch msg.Subject {
				case cfg.NATS.Subjects.MetricsBatch:
					evType = "metrics_batch"
				case cfg.NATS.Subjects.MetricsEpoch:
					evType = "metrics_epoch"
				case cfg.NATS.Subjects.UILog:
					evType = "log"
				default:
					evType = "other"
				}
				w.Write([]byte("event: " + evType + "\n"))
				w.Write([]byte("data: " + string(msg.Data) + "\n\n"))
				flusher.Flush()
				// 缓存
				ring = append(ring, item{Topic: evType, Data: msg.Data, T: time.Now()})
				if len(ring) > 200 {
					ring = ring[len(ring)-200:]
				}
			}
			if r.Context().Err() != nil {
				return
			}
		}
	})

	// 静态 UI
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		b, _ := os.ReadFile("web/index.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(b)
	})
	log.Printf("UI/API listening on %s", cfg.UI.Addr)
	log.Fatal(http.ListenAndServe(cfg.UI.Addr, nil))
}
