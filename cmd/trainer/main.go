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

	// 训练初始化事件
	_ = bus.PublishJSON(cfg.NATS.Subjects.TrainInit,
		natsbus.MsgID("init-", time.Now().UnixNano()),
		events.TrainInit{
			Dataset: cfg.Training.Dataset, Epochs: cfg.Training.Epochs, BatchSize: cfg.Training.BatchSize,
			T: cfg.Training.Timesteps, K: cfg.Training.FixedPointK, Tol: cfg.Training.FixedPointTol,
			Hidden: cfg.Training.Hidden, LR: cfg.Training.LR, Time: events.Now(),
		})

	// 数据
	var loader *data.Loader
	loader, _ = data.NewLoaderMNIST(cfg.Training.DataRoot, cfg.Training.BatchSize, cfg.Training.Seed)

	// 模型
	net := snn.NewThreeCompNet(
		cfg.Model.Input, cfg.Training.Hidden, cfg.Model.Output, cfg.Training.Seed,
		cfg.Training.Theta, cfg.Training.AlphaB, cfg.Training.AlphaA, cfg.Training.AlphaS,
		cfg.Training.KappaBS, cfg.Training.KappaAS,
	)

	// 训练循环
	globalStep := 0
	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		// FPT 的 K 次近似迭代（这里用“外层 K 循环 + 残差估计”来发事件；真正的固定点求解可按需替换）
		for k := 1; k <= cfg.Training.FixedPointK; k++ {
			residual := rand.Float64() * 0.01 // 演示：可替换为真实固定点残差
			_ = bus.PublishJSON(cfg.NATS.Subjects.TrainIter,
				natsbus.MsgID("fpt-", epoch, "-", k, "-", time.Now().UnixNano()),
				events.FPTRound{Epoch: epoch, Step: globalStep, K: k, Residual: residual, Time: events.Now()},
			)
		}

		// 遍历 batch
		var sumLoss, sumAcc float64
		var steps int
		for {
			b, ok := loader.Next()
			if !ok {
				break
			}
			steps++
			globalStep++

			// 前向（固定 T）
			logits, cache := net.Forward(b.X, cfg.Training.Timesteps)
			// 只训练读出层（可扩展端到端）
			loss, acc := net.BackpropReadout(cache, b.X, logits, b.Y, cfg.Training.LR)

			sumLoss += loss
			sumAcc += acc

			// 批次指标事件
			_ = bus.PublishJSON(cfg.NATS.Subjects.MetricsBatch,
				natsbus.MsgID("mb-", epoch, "-", steps),
				events.MetricsBatch{Epoch: epoch, Step: steps, Loss: loss, Acc: acc, Time: events.Now()},
			)

			// UI log
			_ = bus.PublishJSON(cfg.NATS.Subjects.UILog,
				natsbus.MsgID("log-", epoch, "-", steps),
				events.UISysLog{Level: "INFO",
					Msg:  fmt.Sprintf("epoch=%d step=%d loss=%.4f acc=%.4f", epoch, steps, loss, acc),
					Time: events.Now(),
				},
			)
		}

		// epoch 汇总
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
