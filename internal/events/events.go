package events

import "time"

type TrainInit struct {
	Dataset   string  `json:"dataset"`
	Epochs    int     `json:"epochs"`
	BatchSize int     `json:"batch_size"`
	T         int     `json:"timesteps"`
	K         int     `json:"fixed_point_K"`
	Tol       float64 `json:"fixed_point_tol"`
	Hidden    int     `json:"hidden"`
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
	Epoch int     `json:"epoch"`
	Step  int     `json:"step"`
	Loss  float64 `json:"loss"`
	Acc   float64 `json:"acc"`
	Time  int64   `json:"time_unix"`
}

type MetricsEpoch struct {
	Epoch int     `json:"epoch"`
	Loss  float64 `json:"loss"`
	Acc   float64 `json:"acc"`
	Time  int64   `json:"time_unix"`
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

func Now() int64 { return time.Now().Unix() }
