package trainer

import (
	"math"
	"sort"

	"github.com/imwower/snn-go/internal/events"
	"github.com/imwower/snn-go/internal/snn"
)

const (
	defaultMaxSpikeTotal    = 256
	defaultMaxSpikePerLayer = 64
)

// BuildSpikeBursts converts the cached spike activations into a compact per-layer summary suitable for UI streaming.
func BuildSpikeBursts(cache snn.ForwardCache, hidden, layers, maxTotal, maxPerLayer int) []events.SpikeBurst {
	if hidden <= 0 {
		return nil
	}
	if layers <= 0 {
		layers = 1
	}
	if maxTotal <= 0 {
		maxTotal = defaultMaxSpikeTotal
	}
	if maxPerLayer <= 0 {
		maxPerLayer = defaultMaxSpikePerLayer
	}

	counts := make([]int, hidden)
	for _, frame := range cache.S {
		for idx, val := range frame {
			if val > 0 {
				counts[idx%hidden]++
			}
		}
	}

	type entry struct {
		idx   int
		count int
	}
	entries := make([]entry, 0, hidden)
	totalCount := 0
	for idx, cnt := range counts {
		if cnt <= 0 {
			continue
		}
		entries = append(entries, entry{idx: idx, count: cnt})
		totalCount += cnt
	}
	if len(entries) == 0 {
		return nil
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].count == entries[j].count {
			return entries[i].idx < entries[j].idx
		}
		return entries[i].count > entries[j].count
	})

	if len(entries) > maxTotal {
		entries = entries[:maxTotal]
	}

	perLayer := int(math.Ceil(float64(hidden) / float64(layers)))
	if perLayer <= 0 {
		perLayer = hidden
	}

	layerBuckets := make(map[int][]int, layers)
	layerTotals := make(map[int]int, layers)

	for _, e := range entries {
		layer := e.idx / perLayer
		if layer >= layers {
			layer = layers - 1
		}
		local := e.idx - layer*perLayer
		current := layerBuckets[layer]
		if len(current) >= maxPerLayer {
			continue
		}
		layerBuckets[layer] = append(current, local)
		layerTotals[layer] += e.count
	}

	if len(layerBuckets) == 0 {
		return nil
	}

	results := make([]events.SpikeBurst, 0, len(layerBuckets))
	for layer, neurons := range layerBuckets {
		sort.Ints(neurons)
		burst := events.SpikeBurst{
			Layer:   layer,
			Neurons: neurons,
		}
		if totalCount > 0 {
			burst.Power = float64(layerTotals[layer]) / float64(totalCount)
		}
		results = append(results, burst)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Layer < results[j].Layer
	})
	return results
}
