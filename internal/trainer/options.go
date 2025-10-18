package trainer

// Options describes a single training run configuration.
type Options struct {
	Dataset       string
	DataRoot      string
	Epochs        int
	BatchSize     int
	Timesteps     int
	FixedPointK   int
	FixedPointTol float64
	LearningRate  float64
	Hidden        int
	Mode          string
	Layers        int
	NetworkSize   int
	Seed          int64
	EndToEnd      bool
	InputSize     int
	OutputSize    int
	KappaBS       float64
	KappaAS       float64
	Theta         float64
	AlphaB        float64
	AlphaA        float64
	AlphaS        float64
}

// Validate ensures the options contain sane values; zero values are considered invalid for required fields.
func (o Options) Validate() bool {
	switch {
	case o.Dataset == "":
		return false
	case o.DataRoot == "":
		return false
	case o.Epochs <= 0:
		return false
	case o.BatchSize <= 0:
		return false
	case o.Timesteps <= 0:
		return false
	case o.FixedPointK <= 0:
		return false
	case o.FixedPointTol <= 0:
		return false
	case o.LearningRate <= 0:
		return false
	case o.Hidden <= 0:
		return false
	case o.InputSize <= 0:
		return false
	case o.OutputSize <= 0:
		return false
	default:
		return true
	}
}
