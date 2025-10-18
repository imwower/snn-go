package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/data"
	"github.com/imwower/snn-go/internal/events"
	"github.com/imwower/snn-go/internal/natsbus"
	"github.com/imwower/snn-go/internal/snn"
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

	// init event
	_ = bus.PublishJSON(cfg.NATS.Subjects.TrainInit,
		natsbus.MsgID("init-", time.Now().UnixNano()),
		events.TrainInit{
			Dataset: cfg.Training.Dataset, Epochs: cfg.Training.Epochs, BatchSize: cfg.Training.BatchSize,
			T: cfg.Training.Timesteps, K: cfg.Training.FixedPointK, Tol: cfg.Training.FixedPointTol,
			Hidden: cfg.Training.Hidden, LR: cfg.Training.LR, Time: events.Now(),
		})

	ld, _ := data.NewLoaderMNIST(cfg.Training.DataRoot, cfg.Training.BatchSize, cfg.Training.Seed)

	net := snn.NewThreeCompNet(
		cfg.Model.Input, cfg.Training.Hidden, cfg.Model.Output, cfg.Training.Seed,
		cfg.Training.Theta, cfg.Training.AlphaB, cfg.Training.AlphaA, cfg.Training.AlphaS,
		cfg.Training.KappaBS, cfg.Training.KappaAS,
	)

	globalStep := 0
	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		// 固定点近似迭代（演示：随机残差；如需严格可计算 ||v^{t+1}-v^t||/||v^t||）
		for k := 1; k <= cfg.Training.FixedPointK; k++ {
			residual := rand.Float64() * 0.01
			_ = bus.PublishJSON(cfg.NATS.Subjects.TrainIter,
				natsbus.MsgID("fpt-", epoch, "-", k, "-", time.Now().UnixNano()),
				events.FPTRound{Epoch: epoch, Step: globalStep, K: k, Residual: residual, Time: events.Now()},
			)
		}

		var sumLoss, sumAcc float64
		var steps int
		for {
			b, ok := ld.Next()
			if !ok {
				break
			}
			steps++
			globalStep++

			logits, cache := net.Forward(b.X, cfg.Training.Timesteps)
			loss, acc := net.BackpropReadout(cache, b.X, logits, b.Y, cfg.Training.LR)

			sumLoss += loss
			sumAcc += acc

			_ = bus.PublishJSON(cfg.NATS.Subjects.MetricsBatch,
				natsbus.MsgID("mb-", epoch, "-", steps),
				events.MetricsBatch{Epoch: epoch, Step: steps, Loss: loss, Acc: acc, Time: events.Now()},
			)

			_ = bus.PublishJSON(cfg.NATS.Subjects.UILog,
				natsbus.MsgID("log-", epoch, "-", steps),
				events.UISysLog{
					Level: "INFO",
					Msg:   fmt.Sprintf("epoch=%d step=%d loss=%.4f acc=%.4f", epoch, steps, loss, acc),
					Time:  events.Now(),
				},
			)
		}
		epochLoss := sumLoss / float64(steps)
		epochAcc := sumAcc / float64(steps)

		_ = bus.PublishJSON(cfg.NATS.Subjects.MetricsEpoch,
			natsbus.MsgID("me-", epoch),
			events.MetricsEpoch{Epoch: epoch, Loss: epochLoss, Acc: epochAcc, Time: events.Now()},
		)
		_ = bus.PublishJSON(cfg.NATS.Subjects.ParamsApply,
			natsbus.MsgID("pa-", epoch),
			events.ParamApply{Epoch: epoch, Step: steps, LR: cfg.Training.LR, Time: events.Now()},
		)
	}

	_ = bus.PublishJSON(cfg.NATS.Subjects.UILog,
		natsbus.MsgID("log-done-", time.Now().UnixNano()),
		events.UISysLog{Level: "INFO", Msg: "training finished", Time: events.Now()},
	)
}
