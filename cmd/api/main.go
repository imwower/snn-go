package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/datasets"
	"github.com/imwower/snn-go/internal/events"
	"github.com/nats-io/nats.go"
)

type sseHub struct {
	mu      sync.RWMutex
	clients map[chan []byte]struct{}
}

func newHub() *sseHub {
	return &sseHub{
		clients: make(map[chan []byte]struct{}),
	}
}

func (h *sseHub) add(ch chan []byte) {
	h.mu.Lock()
	h.clients[ch] = struct{}{}
	h.mu.Unlock()
}

func (h *sseHub) del(ch chan []byte) {
	h.mu.Lock()
	delete(h.clients, ch)
	h.mu.Unlock()
}

func (h *sseHub) broadcast(evType string, payload []byte) {
	msg := append([]byte("event: "+evType+"\n"+"data: "), payload...)
	msg = append(msg, []byte("\n\n")...)
	h.mu.RLock()
	for ch := range h.clients {
		select {
		case ch <- msg:
		default:
		}
	}
	h.mu.RUnlock()
}

func sseHandler(h *sseHub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "SSE not supported", http.StatusInternalServerError)
			return
		}

		ch := make(chan []byte, 256)
		h.add(ch)
		defer h.del(ch)

		notify := r.Context().Done()
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case msg := <-ch:
				if _, err := w.Write(msg); err != nil {
					return
				}
				flusher.Flush()
			case <-ticker.C:
				if _, err := w.Write([]byte(": ping\n\n")); err != nil {
					return
				}
				flusher.Flush()
			case <-notify:
				return
			}
		}
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetPrefix("[API] ")

	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("加载配置失败：%v", err)
	}
	log.Printf("监听 %s；NATS=%s；Stream=%s", cfg.UI.Addr, cfg.NATS.URL, cfg.NATS.Stream)

	http.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(cfg)
	})

	hub := newHub()
	setCORS := func(w http.ResponseWriter, methods string) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", methods)
	}
	replyJSON := func(w http.ResponseWriter, status int, payload any) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(payload)
	}
	broadcastLog := func(level, msg string) {
		payload, _ := json.Marshal(events.UISysLog{
			Level: level,
			Msg:   msg,
			Time:  events.Now(),
		})
		hub.broadcast("log", payload)
	}

	http.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		sseHandler(hub)(w, r)
	})

	dsManager := datasets.NewManager(cfg.Training.DataRoot, func(event string, payload []byte) {
		hub.broadcast(event, payload)
	})
	listHandler := func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet, http.MethodOptions:
			dsManager.HandleList(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
	http.HandleFunc("/api/datasets", listHandler)
	http.HandleFunc("/api/datasets/", listHandler)

	downloadHandler := func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost, http.MethodOptions:
			dsManager.HandleDownload(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
	http.HandleFunc("/api/datasets/download", downloadHandler)
	http.HandleFunc("/api/datasets/download/", downloadHandler)

	type trainInitRequest struct {
		Dataset     string  `json:"dataset"`
		Mode        string  `json:"mode"`
		NetworkSize int     `json:"network_size"`
		Layers      int     `json:"layers"`
		LR          float64 `json:"lr"`
		K           int     `json:"K"`
		Tol         float64 `json:"tol"`
		T           int     `json:"T"`
	}

	http.HandleFunc("/api/train/init", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodOptions:
			setCORS(w, "POST, OPTIONS")
			w.WriteHeader(http.StatusNoContent)
			return
		case http.MethodPost:
			setCORS(w, "POST, OPTIONS")
			var req trainInitRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				return
			}
			if req.Dataset == "" {
				http.Error(w, "dataset required", http.StatusBadRequest)
				return
			}
			msg := fmt.Sprintf("初始化训练：dataset=%s mode=%s lr=%.4f layers=%d size=%d T=%d K=%d tol=%.6f",
				req.Dataset, req.Mode, req.LR, req.Layers, req.NetworkSize, req.T, req.K, req.Tol)
			broadcastLog("INFO", msg)

			initPayload := events.TrainInit{
				Dataset:   req.Dataset,
				Epochs:    cfg.Training.Epochs,
				BatchSize: cfg.Training.BatchSize,
				T:         req.T,
				K:         req.K,
				Tol:       req.Tol,
				Hidden:    cfg.Training.Hidden,
				LR:        req.LR,
				Time:      events.Now(),
			}
			payload, _ := json.Marshal(initPayload)
			hub.broadcast("train_init", payload)

			replyJSON(w, http.StatusAccepted, map[string]string{"status": "ok"})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	http.HandleFunc("/api/train/start", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodOptions:
			setCORS(w, "POST, OPTIONS")
			w.WriteHeader(http.StatusNoContent)
			return
		case http.MethodPost:
			setCORS(w, "POST, OPTIONS")
			broadcastLog("INFO", "启动训练流程")
			replyJSON(w, http.StatusAccepted, map[string]string{"status": "ok"})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	http.HandleFunc("/api/train/stop", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodOptions:
			setCORS(w, "POST, OPTIONS")
			w.WriteHeader(http.StatusNoContent)
			return
		case http.MethodPost:
			setCORS(w, "POST, OPTIONS")
			broadcastLog("WARNING", "收到停止训练指令")
			replyJSON(w, http.StatusAccepted, map[string]string{"status": "ok"})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	nc, err := nats.Connect(cfg.NATS.URL)
	if err != nil {
		log.Fatalf("NATS 连接失败：%v", err)
	}
	js, err := nc.JetStream()
	if err != nil {
		log.Fatalf("JetStream 初始化失败：%v", err)
	}

	type subSpec struct {
		Subject string
		Durable string
		Event   string
	}
	specs := []subSpec{
		{cfg.NATS.Subjects.MetricsBatch, "SSE_BATCH", "metrics_batch"},
		{cfg.NATS.Subjects.MetricsEpoch, "SSE_EPOCH", "metrics_epoch"},
		{cfg.NATS.Subjects.UILog, "SSE_LOG", "log"},
		{cfg.NATS.Subjects.TrainInit, "SSE_INIT", "train_init"},
		{cfg.NATS.Subjects.TrainIter, "SSE_ITER", "train_iter"},
	}

	for _, sp := range specs {
		sub, err := js.PullSubscribe(sp.Subject, sp.Durable, nats.BindStream(cfg.NATS.Stream))
		if err != nil {
			log.Fatalf("订阅失败 subject=%s durable=%s：%v", sp.Subject, sp.Durable, err)
		}
		go func(s *nats.Subscription, ev string) {
			for {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				msgs, err := s.Fetch(128, nats.Context(ctx))
				cancel()
				if err != nil {
					if errors.Is(err, context.DeadlineExceeded) || err == nats.ErrTimeout {
						time.Sleep(200 * time.Millisecond)
						continue
					}
					log.Printf("拉取失败（%s）：%v", ev, err)
					time.Sleep(500 * time.Millisecond)
					continue
				}
				for _, m := range msgs {
					hub.broadcast(ev, m.Data)
					_ = m.Ack()
				}
				if len(msgs) == 0 {
					time.Sleep(200 * time.Millisecond)
				}
			}
		}(sub, sp.Event)
	}

	dist := "ui-vue/dist"
	index := dist + "/index.html"
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if _, err := os.Stat(index); err == nil {
			http.FileServer(http.Dir(dist)).ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte("请先构建前端：cd ui-vue && npm i && npm run build\n"))
	})

	log.Printf("SSE/API 已启动，地址 %s", cfg.UI.Addr)
	log.Fatal(http.ListenAndServe(cfg.UI.Addr, nil))
}
