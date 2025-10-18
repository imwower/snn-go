package main

import (
	"fmt"
	"log"
	"math"
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
		log.Fatalf("加载配置失败：%v", err)
	}
	bus, err := natsbus.Connect(natsbus.StreamConfig{
		Stream: cfg.NATS.Stream, URL: cfg.NATS.URL, DupeWindowSec: cfg.NATS.DupeWindowSec,
	})
	if err != nil {
		log.Fatalf("连接 NATS 失败：%v", err)
	}
	defer bus.Close()

	// 初始化事件
	_ = bus.PublishJSON(cfg.NATS.Subjects.TrainInit,
		natsbus.MsgID("init-", time.Now().UnixNano()),
		events.TrainInit{
			Dataset: cfg.Training.Dataset, Epochs: cfg.Training.Epochs, BatchSize: cfg.Training.BatchSize,
			T: cfg.Training.Timesteps, K: cfg.Training.FixedPointK, Tol: cfg.Training.FixedPointTol,
			Hidden: cfg.Training.Hidden, LR: cfg.Training.LR, Time: events.Now(),
		})

	// 数据集：MNIST/FASHION/SYNTH 自动选择
	ld, err := data.NewLoader(cfg.Training.Dataset, cfg.Training.DataRoot, cfg.Training.BatchSize, cfg.Training.Seed)
	if err != nil {
		log.Printf("数据加载器异常：%v", err)
	}

	// 模型
	net := snn.NewThreeCompNet(
		cfg.Model.Input, cfg.Training.Hidden, cfg.Model.Output, cfg.Training.Seed,
		cfg.Training.Theta, cfg.Training.AlphaB, cfg.Training.AlphaA, cfg.Training.AlphaS,
		cfg.Training.KappaBS, cfg.Training.KappaAS,
	)

	globalStep := 0
	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {

		var sumLoss, sumAcc float64
		var steps int

		for {
			b, ok := ld.Next()
			if !ok {
				break
			}
			steps++
			globalStep++

			// —— 固定时间步前向 —— //
			logits, cache := net.Forward(b.X, cfg.Training.Timesteps)

			// —— FPT 残差（真实计算）：以 Vs 时间序列做 ||v^t - v^{t-1}|| / (||v^{t-1}||+ε) 的 batch 均值 —— //
			residual := residualFromCache(cache)

			_ = bus.PublishJSON(cfg.NATS.Subjects.TrainIter,
				natsbus.MsgID("fpt-", epoch, "-", steps, "-", time.Now().UnixNano()),
				events.FPTRound{Epoch: epoch, Step: steps, K: 1, Residual: residual, Time: events.Now()},
			)

			// —— 反向（默认仅读出层） —— //
			var loss, acc float64
			if cfg.Training.EndToEnd {
				loss, acc = net.BackpropFullSTE(cache, b.X, logits, b.Y, cfg.Training.LR)
			} else {
				loss, acc = net.BackpropReadout(cache, b.X, logits, b.Y, cfg.Training.LR)
			}

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
					Msg:   fmt.Sprintf("轮次=%d 步数=%d 损失=%.4f 准确率=%.4f 残差=%.6f", epoch, steps, loss, acc, residual),
					Time:  events.Now(),
				},
			)
		}

		epochLoss := sumLoss / math.Max(1, float64(steps))
		epochAcc := sumAcc / math.Max(1, float64(steps))

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
		events.UISysLog{Level: "INFO", Msg: "训练已完成", Time: events.Now()},
	)
}

// residualFromCache：步间差分残差（FPT 近似）
// r = mean_{t=1..T-1} ||v_s^t - v_s^{t-1}||_2 / (||v_s^{t-1}||_2 + 1e-8)
// 若已保留 F(v)-v，可替换为函数残差测度，两者在离散一阶迭代下等价。
func residualFromCache(c snn.ForwardCache) float64 {
	if len(c.Vs) <= 1 {
		return 0
	}
	T := len(c.Vs)
	BH := len(c.Vs[0])
	var sum float64
	var cnt int
	for t := 1; t < T; t++ {
		var num, den float64
		for i := 0; i < BH; i++ {
			d := c.Vs[t][i] - c.Vs[t-1][i]
			num += d * d
			den += c.Vs[t-1][i] * c.Vs[t-1][i]
		}
		r := math.Sqrt(num) / (math.Sqrt(den) + 1e-8)
		sum += r
		cnt++
	}
	if cnt == 0 {
		return 0
	}
	return sum / float64(cnt)
}
