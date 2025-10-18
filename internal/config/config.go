package config

import (
	"encoding/json"
	"errors"
	"os"
)

type Subjects struct {
	MetricsBatch string `json:"metrics_batch"`
	MetricsEpoch string `json:"metrics_epoch"`
	TrainInit    string `json:"train_init"`
	TrainIter    string `json:"train_iter"`
	ParamsApply  string `json:"params_apply"`
	ParamsSnap   string `json:"params_snap"`
	UILog        string `json:"ui_log"`
}

type NATS struct {
	URL           string   `json:"url"`
	Stream        string   `json:"stream"`
	Subjects      Subjects `json:"subjects"`
	DupeWindowSec int      `json:"dupe_window_sec"`
}

type Training struct {
	Dataset       string  `json:"dataset"`
	DataRoot      string  `json:"data_root"`
	Epochs        int     `json:"epochs"`
	BatchSize     int     `json:"batch_size"`
	Timesteps     int     `json:"timesteps"`
	FixedPointK   int     `json:"fixed_point_K"`
	FixedPointTol float64 `json:"fixed_point_tol"`
	LR            float64 `json:"lr"`
	Theta         float64 `json:"theta"`
	AlphaB        float64 `json:"alpha_b"`
	AlphaA        float64 `json:"alpha_a"`
	AlphaS        float64 `json:"alpha_s"`
	KappaBS       float64 `json:"kappa_bs"`
	KappaAS       float64 `json:"kappa_as"`
	Hidden        int     `json:"hidden"`
	Seed          int64   `json:"seed"`
}

type Model struct {
	Input  int `json:"input"`
	Output int `json:"output"`
}

type UI struct {
	Addr string `json:"addr"`
}

type Config struct {
	NATS     NATS     `json:"nats"`
	Training Training `json:"training"`
	Model    Model    `json:"model"`
	UI       UI       `json:"ui"`
}

func Load(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, errors.New("failed to parse config.yaml as JSON: " + err.Error())
	}
	return cfg, nil
}
