package natsbus

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
)

type Bus struct {
	nc *nats.Conn
	js nats.JetStreamContext
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
		nc.Close()
		return nil, err
	}
	streamCfg := &nats.StreamConfig{
		Name:       cfg.Stream,
		Subjects:   []string{"snn.>"},
		Storage:    nats.FileStorage,
		Retention:  nats.LimitsPolicy,
		Duplicates: time.Duration(cfg.DupeWindowSec) * time.Second,
	}
	if _, err := js.AddStream(streamCfg); err != nil {
		if errors.Is(err, nats.ErrStreamNameAlreadyInUse) {
			info, infoErr := js.StreamInfo(cfg.Stream)
			if infoErr != nil {
				nc.Close()
				return nil, fmt.Errorf("natsbus: describe stream %s: %w", cfg.Stream, infoErr)
			}
			dupWindow := time.Duration(cfg.DupeWindowSec) * time.Second
			subjectMatch := len(info.Config.Subjects) == 1 && info.Config.Subjects[0] == "snn.>"
			storageMatch := info.Config.Storage == nats.FileStorage
			retentionMatch := info.Config.Retention == nats.LimitsPolicy
			dupMatch := info.Config.Duplicates == dupWindow
			if !subjectMatch || !storageMatch || !retentionMatch || !dupMatch {
				updateCfg := info.Config
				updateCfg.Subjects = []string{"snn.>"}
				updateCfg.Storage = nats.FileStorage
				updateCfg.Retention = nats.LimitsPolicy
				updateCfg.Duplicates = dupWindow
				if _, err := js.UpdateStream(&updateCfg); err != nil {
					nc.Close()
					return nil, fmt.Errorf("natsbus: update stream %s: %w", cfg.Stream, err)
				}
			}
		} else {
			nc.Close()
			return nil, fmt.Errorf("natsbus: add stream %s: %w", cfg.Stream, err)
		}
	}
	return &Bus{nc: nc, js: js}, nil
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
