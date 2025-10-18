package natsbus

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
)

type Bus struct {
	nc  *nats.Conn
	js  nats.JetStreamContext
	cfg StreamConfig
}

type StreamConfig struct {
	Stream        string
	URL           string
	DupeWindowSec int
}

func Connect(cfg StreamConfig) (*Bus, error) {
	nc, err := nats.Connect(cfg.URL)
	if err != nil {
		return nil, err
	}
	js, err := nc.JetStream()
	if err != nil {
		return nil, err
	}
	// 幂等 + 至少一次：创建流（若已存在忽略）
	_, _ = js.AddStream(&nats.StreamConfig{
		Name:       cfg.Stream,
		Subjects:   []string{"snn.>"},
		Storage:    nats.FileStorage,
		Retention:  nats.LimitsPolicy,
		Duplicates: time.Duration(cfg.DupeWindowSec) * time.Second,
	})
	return &Bus{nc: nc, js: js, cfg: cfg}, nil
}

func (b *Bus) PublishJSON(subject string, msgID string, v any) error {
	bts, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = b.js.PublishMsg(&nats.Msg{
		Subject: subject,
		Header:  nats.Header{"Nats-Msg-Id": []string{msgID}},
		Data:    bts,
	})
	return err
}

func (b *Bus) Subscribe(subject, durable string, cb func(m *nats.Msg)) (*nats.Subscription, error) {
	return b.js.PullSubscribe(subject, durable, nats.BindStream(b.cfg.Stream))
}

func Ack(msg *nats.Msg) { _ = msg.Ack() }

func (b *Bus) FetchAndHandle(ctx context.Context, sub *nats.Subscription, batch int, cb func(*nats.Msg)) error {
	msgs, err := sub.Fetch(batch, nats.Context(ctx))
	if err != nil {
		return err
	}
	for _, m := range msgs {
		cb(m)
	}
	return nil
}

func (b *Bus) Close() { b.nc.Drain(); b.nc.Close() }

func MsgID(parts ...any) string {
	return fmt.Sprint(parts...) // 简约组合（可加入时间戳/uuid）
}
