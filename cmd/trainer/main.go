package main

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/data"
	"github.com/imwower/snn-go/internal/events"
	"github.com/imwower/snn-go/internal/natsbus"
	"github.com/imwower/snn-go/internal/snn"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetPrefix("[TRAIN] ")

	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("加载配置失败：%v", err)
	}

	log.Printf("系统基线：NumCPU=%d GOMAXPROCS=%d", runtime.NumCPU(), runtime.GOMAXPROCS(0))

	bus, err := natsbus.Connect(natsbus.StreamConfig{
		Stream:        cfg.NATS.Stream,
		URL:           cfg.NATS.URL,
		DupeWindowSec: cfg.NATS.DupeWindowSec,
	})
	if err != nil {
		log.Fatalf("NATS 连接失败：%v", err)
	}
	defer bus.Close()

	_ = bus.PublishJSON(cfg.NATS.Subjects.TrainInit,
		natsbus.MsgID("init-", time.Now().UnixNano()),
		events.TrainInit{
			Dataset: cfg.Training.Dataset, Epochs: cfg.Training.Epochs, BatchSize: cfg.Training.BatchSize,
			T: cfg.Training.Timesteps, K: cfg.Training.FixedPointK, Tol: cfg.Training.FixedPointTol,
			Hidden: cfg.Training.Hidden, LR: cfg.Training.LR, Time: events.Now(),
		},
	)

	ld, _ := data.NewLoader(cfg.Training.Dataset, cfg.Training.DataRoot, cfg.Training.BatchSize, cfg.Training.Seed)
	net := snn.NewThreeCompNet(cfg.Model.Input, cfg.Training.Hidden, cfg.Model.Output, cfg.Training.Seed,
		cfg.Training.Theta, cfg.Training.AlphaB, cfg.Training.AlphaA, cfg.Training.AlphaS,
		cfg.Training.KappaBS, cfg.Training.KappaAS,
	)

	var globalExamples int64
	bestAcc := -1.0
	bestLoss := math.MaxFloat64

	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		if ld != nil {
			ld.Reset(cfg.Training.Seed + int64(epoch))
		}

		epochStart := time.Now()
		var sumLoss, sumAcc, sumTPS float64
		var steps int
		emaLoss, emaAcc := 0.0, 0.0
		const emaAlpha = 0.1

		log.Printf("=== 开始第 %d/%d 轮 ===", epoch, cfg.Training.Epochs)

		for {
			batch, ok := ld.Next()
			if !ok {
				break
			}
			steps++

			stepStart := time.Now()
			logits, cache := net.Forward(batch.X, cfg.Training.Timesteps)

			residual := residualFromVs(cache)
			loss, acc := net.BackpropReadout(cache, batch.X, logits, batch.Y, cfg.Training.LR)
			top5 := topKAcc(logits, batch.Y, 5)

			stepMS := float64(time.Since(stepStart).Milliseconds())
			if stepMS <= 0 {
				stepMS = 1
			}
			tps := float64(len(batch.X)) / (stepMS / 1000.0)

			globalExamples += int64(len(batch.X))
			emaLoss = emaAlpha*loss + (1-emaAlpha)*emaLoss
			emaAcc = emaAlpha*acc + (1-emaAlpha)*emaAcc

			sumLoss += loss
			sumAcc += acc
			sumTPS += tps

			_ = bus.PublishJSON(cfg.NATS.Subjects.TrainIter,
				natsbus.MsgID("fpt-", epoch, "-", steps, "-", time.Now().UnixNano()),
				events.FPTRound{Epoch: epoch, Step: steps, K: 1, Residual: residual, Time: events.Now()},
			)

			_ = bus.PublishJSON(cfg.NATS.Subjects.MetricsBatch,
				natsbus.MsgID("mb-", epoch, "-", steps),
				events.MetricsBatch{
					Epoch: epoch, Step: steps,
					Loss: loss, Acc: acc, Top5: top5,
					EMALoss: emaLoss, EMAAcc: emaAcc,
					Throughput: tps, StepMS: stepMS,
					Residual: residual, Examples: globalExamples,
					LR: cfg.Training.LR, Time: events.Now(),
				},
			)

			_ = bus.PublishJSON(cfg.NATS.Subjects.UILog,
				natsbus.MsgID("log-", epoch, "-", steps),
				events.UISysLog{
					Level: "INFO",
					Msg: fmt.Sprintf("epoch=%d step=%d loss=%.4f acc=%.4f top5=%.4f ema_loss=%.4f ema_acc=%.4f tps=%.1f step_ms=%.0f residual=%.6f examples=%d",
						epoch, steps, loss, acc, top5, emaLoss, emaAcc, tps, stepMS, residual, globalExamples),
					Time: events.Now(),
				},
			)
		}

		epochLoss := sumLoss / math.Max(1, float64(steps))
		epochAcc := sumAcc / math.Max(1, float64(steps))
		avgTPS := sumTPS / math.Max(1, float64(steps))
		epochSec := time.Since(epochStart).Seconds()

		if epochAcc > bestAcc {
			bestAcc = epochAcc
		}
		if epochLoss < bestLoss {
			bestLoss = epochLoss
		}

		_ = bus.PublishJSON(cfg.NATS.Subjects.MetricsEpoch,
			natsbus.MsgID("me-", epoch),
			events.MetricsEpoch{
				Epoch: epoch, Loss: epochLoss, Acc: epochAcc,
				BestLoss: bestLoss, BestAcc: bestAcc,
				AvgThroughput: avgTPS, EpochSec: epochSec,
				Time: events.Now(),
			},
		)
		_ = bus.PublishJSON(cfg.NATS.Subjects.ParamsApply,
			natsbus.MsgID("pa-", epoch),
			events.ParamApply{Epoch: epoch, Step: steps, LR: cfg.Training.LR, Time: events.Now()},
		)

		log.Printf("=== 结束第 %d 轮：loss=%.4f acc=%.4f avg_tps=%.1f time=%.1fs ===",
			epoch, epochLoss, epochAcc, avgTPS, epochSec)
	}

	_ = bus.PublishJSON(cfg.NATS.Subjects.UILog,
		natsbus.MsgID("log-done-", time.Now().UnixNano()),
		events.UISysLog{Level: "INFO", Msg: "训练完成", Time: events.Now()},
	)
	log.Println("训练完成，退出")
}

func residualFromVs(c snn.ForwardCache) float64 {
	if len(c.Vs) <= 1 {
		return 0
	}
	T := len(c.Vs)
	BH := len(c.Vs[0])
	var sum float64
	var count int
	for t := 1; t < T; t++ {
		var num, den float64
		for i := 0; i < BH; i++ {
			d := c.Vs[t][i] - c.Vs[t-1][i]
			num += d * d
			den += c.Vs[t-1][i] * c.Vs[t-1][i]
		}
		r := math.Sqrt(num) / (math.Sqrt(den) + 1e-8)
		sum += r
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func topKAcc(logits [][]float64, y []int, k int) float64 {
	B := len(logits)
	if B == 0 {
		return 0
	}
	correct := 0
	for i := 0; i < B; i++ {
		type pair struct {
			idx int
			val float64
		}
		arr := make([]pair, len(logits[i]))
		for j, v := range logits[i] {
			arr[j] = pair{idx: j, val: v}
		}
		for a := 0; a < k && a < len(arr); a++ {
			maxIdx := a
			for b := a + 1; b < len(arr); b++ {
				if arr[b].val > arr[maxIdx].val {
					maxIdx = b
				}
			}
			arr[a], arr[maxIdx] = arr[maxIdx], arr[a]
		}
		found := false
		for a := 0; a < k && a < len(arr); a++ {
			if arr[a].idx == y[i] {
				found = true
				break
			}
		}
		if found {
			correct++
		}
	}
	return float64(correct) / float64(B)
}
