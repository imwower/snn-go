package natsbus

import (
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
	_, _ = js.AddStream(&nats.StreamConfig{
		Name:       cfg.Stream,
		Subjects:   []string{"snn.>"},
		Storage:    nats.FileStorage,
		Retention:  nats.LimitsPolicy,
		Duplicates: time.Duration(cfg.DupeWindowSec) * time.Second,
	})
	return &Bus{nc: nc, js: js, cfg: cfg}, nil
}

func (b *Bus) PublishJSON(subject, msgID string, v any) error {
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

func (b *Bus) Close() {
	_ = b.nc.Drain()
	b.nc.Close()
}

func MsgID(parts ...any) string {
	return fmt.Sprint(parts...)
}
