package trainer

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/data"
	"github.com/imwower/snn-go/internal/events"
	"github.com/imwower/snn-go/internal/natsbus"
	"github.com/imwower/snn-go/internal/snn"
)

// LogFunc is used to stream textual logs to the caller (e.g. SSE).
type LogFunc func(level, message string)

// StatusFunc notifies the caller when the high-level training status changes.
type StatusFunc func(status string)

var (
	ErrRunInProgress   = errors.New("trainer: run already in progress")
	ErrNoRunInProgress = errors.New("trainer: no run in progress")
	ErrInvalidOptions  = errors.New("trainer: invalid options")
)

// Runner owns the lifecycle of a single training loop. It is safe for concurrent use.
type Runner struct {
	cfg      config.Config
	logf     LogFunc
	onStatus StatusFunc

	mu         sync.Mutex
	running    bool
	cancel     context.CancelFunc
	done       chan struct{}
	lastErr    error
	lastStatus string
}

// NewRunner returns a new Runner bound to the provided configuration.
func NewRunner(cfg config.Config, logf LogFunc, onStatus StatusFunc) *Runner {
	if logf == nil {
		logf = func(level, message string) {}
	}
	if onStatus == nil {
		onStatus = func(string) {}
	}
	return &Runner{
		cfg:      cfg,
		logf:     logf,
		onStatus: onStatus,
	}
}

// Start launches a training run with the provided options. Only one run may execute at a time.
func (r *Runner) Start(opts Options) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.running {
		return ErrRunInProgress
	}
	if !opts.Validate() {
		return ErrInvalidOptions
	}

	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel
	r.done = make(chan struct{})
	r.running = true
	r.lastErr = nil
	r.lastStatus = "Training"
	r.onStatus("Training")

	go r.run(ctx, opts)
	return nil
}

// Stop requests the currently running training job to halt. It blocks until the run stops.
func (r *Runner) Stop() error {
	r.mu.Lock()
	if !r.running {
		r.mu.Unlock()
		return ErrNoRunInProgress
	}
	cancel := r.cancel
	done := r.done
	r.mu.Unlock()

	cancel()
	<-done

	r.mu.Lock()
	err := r.lastErr
	r.mu.Unlock()
	return err
}

// Status returns the most recent coarse status string.
func (r *Runner) Status() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.running {
		return "Training"
	}
	if r.lastStatus == "" {
		return "Idle"
	}
	return r.lastStatus
}

// Wait blocks until the current run completes or the provided context finishes. If no run is active it returns immediately.
func (r *Runner) Wait(ctx context.Context) error {
	r.mu.Lock()
	done := r.done
	lastErr := r.lastErr
	running := r.running
	r.mu.Unlock()

	if !running || done == nil {
		return lastErr
	}

	select {
	case <-done:
	case <-ctx.Done():
		return ctx.Err()
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lastErr
}

func (r *Runner) run(ctx context.Context, opts Options) {
	defer func() {
		r.mu.Lock()
		r.running = false
		if errors.Is(r.lastErr, context.Canceled) {
			r.lastStatus = "Stopped"
		} else if r.lastErr != nil {
			r.lastStatus = "Error"
		} else {
			r.lastStatus = "Idle"
		}
		r.mu.Unlock()

		switch r.lastStatus {
		case "Stopped":
			r.onStatus("Stopped")
		case "Error":
			r.onStatus("Error")
		default:
			r.onStatus("Idle")
		}

		close(r.done)
	}()

	logger := log.Default()
	bus, err := natsbus.Connect(natsbus.StreamConfig{
		Stream:        r.cfg.NATS.Stream,
		URL:           r.cfg.NATS.URL,
		DupeWindowSec: r.cfg.NATS.DupeWindowSec,
	})
	if err != nil {
		r.reportError(fmt.Errorf("连接 NATS 失败: %w", err))
		return
	}
	defer bus.Close()

	trainingLog := func(level, message string) {
		logger.Printf("[TRAIN] %s", message)
		r.logf(level, message)
		_ = bus.PublishJSON(r.cfg.NATS.Subjects.UILog,
			natsbus.MsgID("log-", opts.Dataset, "-", opts.Mode, "-", events.Now()),
			events.UISysLog{Level: level, Msg: message, Time: events.Now()},
		)
	}

	logInfo := func(msg string) { trainingLog("INFO", msg) }
	logWarn := func(msg string) { trainingLog("WARNING", msg) }
	logErr := func(msg string) { trainingLog("ERROR", msg) }

	// Publish init event with requested options.
	initMsg := events.TrainInit{
		Dataset:   opts.Dataset,
		Epochs:    opts.Epochs,
		BatchSize: opts.BatchSize,
		T:         opts.Timesteps,
		K:         opts.FixedPointK,
		Tol:       opts.FixedPointTol,
		Hidden:    opts.Hidden,
		LR:        opts.LearningRate,
		Time:      events.Now(),
	}
	_ = bus.PublishJSON(r.cfg.NATS.Subjects.TrainInit,
		natsbus.MsgID("init-", opts.Dataset, "-", events.Now()),
		initMsg,
	)

	logInfo(fmt.Sprintf("开始训练：dataset=%s epochs=%d batch=%d T=%d K=%d lr=%.4f",
		opts.Dataset, opts.Epochs, opts.BatchSize, opts.Timesteps, opts.FixedPointK, opts.LearningRate))

	loader, err := data.NewLoader(opts.Dataset, opts.DataRoot, opts.BatchSize, opts.Seed)
	if err != nil {
		r.reportError(fmt.Errorf("数据加载器异常: %w", err))
		logErr(fmt.Sprintf("数据集 %s 加载失败: %v", opts.Dataset, err))
		return
	}

	net := snn.NewThreeCompNet(
		opts.InputSize, opts.Hidden, opts.OutputSize, opts.Seed,
		opts.Theta, opts.AlphaB, opts.AlphaA, opts.AlphaS,
		opts.KappaBS, opts.KappaAS,
	)

	var globalExamples int64
	bestAcc := -1.0
	bestLoss := math.MaxFloat64

loopEpochs:
	for epoch := 1; epoch <= opts.Epochs; epoch++ {
		select {
		case <-ctx.Done():
			logWarn("收到停止训练请求，准备停止")
			r.lastErr = ctx.Err()
			break loopEpochs
		default:
		}

		epochStart := time.Now()
		var sumLoss, sumAcc, sumTPS float64
		var steps int
		emaLoss, emaAcc := 0.0, 0.0
		const emaAlpha = 0.1
		logInfo(fmt.Sprintf("=== 开始第 %d/%d 轮 ===", epoch, opts.Epochs))

		for {
			select {
			case <-ctx.Done():
				logWarn("训练已被停止")
				r.lastErr = ctx.Err()
				break loopEpochs
			default:
			}

			batch, ok := loader.Next()
			if !ok {
				break
			}
			steps++

			stepStart := time.Now()
			logits, cache := net.Forward(batch.X, opts.Timesteps)
			residual := residualFromVs(cache)

			_ = bus.PublishJSON(r.cfg.NATS.Subjects.TrainIter,
				natsbus.MsgID("fpt-", epoch, "-", steps, "-", time.Now().UnixNano()),
				events.FPTRound{Epoch: epoch, Step: steps, K: 1, Residual: residual, Time: events.Now()},
			)

			var loss, acc float64
			if opts.EndToEnd {
				loss, acc = net.BackpropFullSTE(cache, batch.X, logits, batch.Y, opts.LearningRate)
			} else {
				loss, acc = net.BackpropReadout(cache, batch.X, logits, batch.Y, opts.LearningRate)
			}
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

			_ = bus.PublishJSON(r.cfg.NATS.Subjects.MetricsBatch,
				natsbus.MsgID("mb-", epoch, "-", steps),
				events.MetricsBatch{
					Epoch: epoch, Step: steps,
					Loss: loss, Acc: acc, Top5: top5,
					EMALoss: emaLoss, EMAAcc: emaAcc,
					Throughput: tps, StepMS: stepMS,
					Residual: residual, Examples: globalExamples,
					LR: opts.LearningRate, Time: events.Now(),
				},
			)

			logInfo(fmt.Sprintf("epoch=%d step=%d loss=%.4f acc=%.4f top5=%.4f ema_loss=%.4f ema_acc=%.4f tps=%.1f step_ms=%.0f residual=%.6f examples=%d",
				epoch, steps, loss, acc, top5, emaLoss, emaAcc, tps, stepMS, residual, globalExamples))
		}

		if r.lastErr == context.Canceled {
			break
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

		_ = bus.PublishJSON(r.cfg.NATS.Subjects.MetricsEpoch,
			natsbus.MsgID("me-", epoch),
			events.MetricsEpoch{
				Epoch: epoch, Loss: epochLoss, Acc: epochAcc,
				BestLoss: bestLoss, BestAcc: bestAcc,
				AvgThroughput: avgTPS, EpochSec: epochSec,
				Time: events.Now(),
			},
		)
		_ = bus.PublishJSON(r.cfg.NATS.Subjects.ParamsApply,
			natsbus.MsgID("pa-", epoch),
			events.ParamApply{Epoch: epoch, Step: steps, LR: opts.LearningRate, Time: events.Now()},
		)
		logInfo(fmt.Sprintf("=== 结束第 %d 轮：loss=%.4f acc=%.4f avg_tps=%.1f time=%.1fs ===",
			epoch, epochLoss, epochAcc, avgTPS, epochSec))
	}

	if r.lastErr == context.Canceled {
		return
	}

	logInfo("训练完成")
	_ = bus.PublishJSON(r.cfg.NATS.Subjects.UILog,
		natsbus.MsgID("log-done-", events.Now()),
		events.UISysLog{Level: "INFO", Msg: "训练完成", Time: events.Now()},
	)
}

func (r *Runner) reportError(err error) {
	r.mu.Lock()
	r.lastErr = err
	r.mu.Unlock()
	r.logf("ERROR", err.Error())
	log.Printf("[TRAIN] %v", err)
}

func residualFromVs(c snn.ForwardCache) float64 {
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
