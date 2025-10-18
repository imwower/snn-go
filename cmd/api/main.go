package main

import (
	"context"
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

type wsEnvelope struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

func main() {
	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	// WebSocket Hub
	hub := newHub()
	go hub.run()

	// /ws：WebSocket 实时推送
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serveWS(hub, w, r)
	})

	// /api/config：提供配置（仅标准库 json）
	http.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(cfg)
	})

	// 最近指标（可选）
	ring := make([]ringItem, 0, 200)
	pushRing := func(topic string, data []byte) {
		ring = append(ring, ringItem{Topic: topic, Data: append([]byte(nil), data...), T: time.Now().Unix()})
		if len(ring) > 200 {
			ring = ring[len(ring)-200:]
		}
	}
	http.HandleFunc("/api/metrics/recent", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(ring)
	})

	// JetStream Durable + Pull + ACK：三个主题分别创建 Durable
	nc, err := nats.Connect(cfg.NATS.URL)
	if err != nil {
		log.Fatalf("nats connect: %v", err)
	}
	js, err := nc.JetStream()
	if err != nil {
		log.Fatalf("jetstream: %v", err)
	}

	type subSpec struct {
		Subj    string
		Durable string
		Typ     string
	}
	specs := []subSpec{
		{cfg.NATS.Subjects.MetricsBatch, "UI_BATCH", "metrics_batch"},
		{cfg.NATS.Subjects.MetricsEpoch, "UI_EPOCH", "metrics_epoch"},
		{cfg.NATS.Subjects.UILog, "UI_LOG", "log"},
	}

	for _, sp := range specs {
		sub, err := js.PullSubscribe(sp.Subj, sp.Durable, nats.BindStream(cfg.NATS.Stream))
		if err != nil {
			log.Fatalf("pull subscribe %s: %v", sp.Subj, err)
		}
		// 每个主题起一个拉取协程
		go func(s *nats.Subscription, typ string) {
			for {
				// 批量拉取 + ACK
				ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
				msgs, err := s.Fetch(64, nats.Context(ctx))
				cancel()
				if err != nil && err != nats.ErrTimeout {
					time.Sleep(500 * time.Millisecond)
					continue
				}
				for _, m := range msgs {
					env := wsEnvelope{Type: typ, Data: json.RawMessage(m.Data)}
					b, _ := json.Marshal(&env)
					hub.broadcast <- b
					pushRing(typ, m.Data)
					_ = m.Ack() // 及时 ACK，避免积压
				}
				if len(msgs) == 0 {
					time.Sleep(200 * time.Millisecond) // 空轮询退避
				}
			}
		}(sub, sp.Typ)
	}

	// 静态前端：ui-vue/dist
	dist := "ui-vue/dist"
	index := dist + "/index.html"
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if _, err := os.Stat(index); err == nil {
			http.FileServer(http.Dir(dist)).ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write([]byte("Build UI first: cd ui-vue && npm i && npm run build\n"))
	})

	log.Printf("WebSocket/API listening on %s", cfg.UI.Addr)
	log.Fatal(http.ListenAndServe(cfg.UI.Addr, nil))
}
