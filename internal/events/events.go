package events

import "time"

func Now() int64 { return time.Now().Unix() }

type TrainInit struct {
	Dataset   string  `json:"dataset"`
	Epochs    int     `json:"epochs"`
	BatchSize int     `json:"batch_size"`
	T         int     `json:"timesteps"`
	K         int     `json:"fixed_point_K"`
	Tol       float64 `json:"fixed_point_tol"`
	Hidden    int     `json:"hidden"`
	Layers    int     `json:"layers,omitempty"`
	LR        float64 `json:"lr"`
	Time      int64   `json:"time_unix"`
}

type FPTRound struct {
	Epoch    int     `json:"epoch"`
	Step     int     `json:"step"`
	K        int     `json:"k"`
	Residual float64 `json:"residual"`
	Time     int64   `json:"time_unix"`
}

type MetricsBatch struct {
	Epoch      int     `json:"epoch"`
	Step       int     `json:"step"`
	Loss       float64 `json:"loss"`
	Acc        float64 `json:"acc"`
	Top5       float64 `json:"top5,omitempty"`
	EMALoss    float64 `json:"ema_loss,omitempty"`
	EMAAcc     float64 `json:"ema_acc,omitempty"`
	Throughput float64 `json:"throughput"` // samples/s
	StepMS     float64 `json:"step_ms"`
	Residual   float64 `json:"residual,omitempty"`
	Examples   int64   `json:"examples"` // cumulative samples
	LR         float64 `json:"lr"`
	Time       int64   `json:"time_unix"`
}

type MetricsEpoch struct {
	Epoch         int     `json:"epoch"`
	Loss          float64 `json:"loss"`
	Acc           float64 `json:"acc"`
	BestLoss      float64 `json:"best_loss"`
	BestAcc       float64 `json:"best_acc"`
	AvgThroughput float64 `json:"avg_throughput"`
	EpochSec      float64 `json:"epoch_sec"`
	Time          int64   `json:"time_unix"`
}

type ParamApply struct {
	Epoch int     `json:"epoch"`
	Step  int     `json:"step"`
	LR    float64 `json:"lr"`
	Time  int64   `json:"time_unix"`
}

type UISysLog struct {
	Level string `json:"level"`
	Msg   string `json:"msg"`
	Time  int64  `json:"time_unix"`
}

type SpikeBurst struct {
	Layer   int      `json:"layer"`
	Time    int64    `json:"time_unix"`
	Neurons []int    `json:"neurons"`
	Edges   [][2]int `json:"edges,omitempty"`
	Power   float64  `json:"power,omitempty"`
}
