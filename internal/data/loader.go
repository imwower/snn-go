package data

import (
	"math/rand"
	"strings"
)

const (
	synthFeatures = 784
	synthClasses  = 10
	synthSamples  = 2048
)

// 统一入口：根据数据集类型选择加载器
func NewLoader(dataset, root string, bs int, seed int64) (*Loader, error) {
	name := strings.ToUpper(strings.TrimSpace(dataset))
	switch name {
	case "MNIST", "FASHION":
		if ld, err := NewLoaderMNIST(root, bs, seed); err == nil {
			return ld, nil
		}
		return synth(bs, seed), nil
	case "SYNTH":
		return synth(bs, seed), nil
	default:
		if ld, err := NewLoaderMNIST(root, bs, seed); err == nil {
			return ld, nil
		}
		return synth(bs, seed), nil
	}
}

func synth(bs int, seed int64) *Loader {
	if bs <= 0 {
		bs = 1
	}
	n := synthSamples
	if n < bs {
		n = bs
	}

	rng := rand.New(rand.NewSource(seed))
	images := make([][]float64, n)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		row := make([]float64, synthFeatures)
		for j := range row {
			row[j] = rng.Float64()
		}
		images[i] = row
		labels[i] = rng.Intn(synthClasses)
	}

	ld := &Loader{
		images: images,
		labels: labels,
		bs:     bs,
		seed:   seed,
	}
	ld.reset()
	return ld
}
