package snn_test

import (
	"math"
	"testing"

	"github.com/imwower/snn-go/internal/snn"
)

func TestReadoutAggregationAndUpdate(t *testing.T) {
	net := snn.NewThreeCompNet(1, 1, 2, 7, 0, 0, 0, 0, 0, 0)
	net.W_in = [][]float64{{0.5}}
	net.W_b2s = [][]float64{{1.0}}
	net.W_a2s = [][]float64{{0.0}}
	net.W_out = [][]float64{{1.0, -1.0}}
	net.B_out = []float64{0.0, 0.0}
	net.Theta = 0
	net.AlphaB = 0
	net.AlphaA = 0
	net.AlphaS = 0
	net.KappaBS = 0
	net.KappaAS = 0

	inputs := [][]float64{{1.0}}
	labels := []int{0}
	logits, cache := net.Forward(inputs, 2)
	if got, want := len(logits), 1; got != want {
		t.Fatalf("unexpected logits batch size: got %d want %d", got, want)
	}
	t.Logf("logits after forward: %+v", logits)
	if diff := math.Abs(logits[0][0] - 2.0); diff > 1e-9 {
		t.Fatalf("unexpected positive logit: diff=%g", diff)
	}
	if diff := math.Abs(logits[0][1] + 2.0); diff > 1e-9 {
		t.Fatalf("unexpected negative logit: diff=%g", diff)
	}

	vsSum := 0.0
	for _, tSlice := range cache.Vs {
		vsSum += tSlice[0]
	}
	maxLogit := logits[0][0]
	if logits[0][1] > maxLogit {
		maxLogit = logits[0][1]
	}
	exp0 := math.Exp(logits[0][0] - maxLogit)
	exp1 := math.Exp(logits[0][1] - maxLogit)
	sumExp := exp0 + exp1
	prob0 := exp0 / sumExp
	prob1 := exp1 / sumExp
	expectedLoss := -math.Log(prob0 + 1e-12)

	loss, acc := net.BackpropReadout(cache, inputs, logits, labels, 0.1)
	t.Logf("loss=%.6f acc=%.3f", loss, acc)
	if diff := math.Abs(loss - expectedLoss); diff > 1e-9 {
		t.Fatalf("unexpected loss: diff=%g", diff)
	}
	if math.Abs(acc-1.0) > 1e-9 {
		t.Fatalf("expected accuracy 1, got %f", acc)
	}

	expectedWOut0 := 1.0 - 0.1*(prob0-1.0)*vsSum
	expectedWOut1 := -1.0 - 0.1*(prob1-0.0)*vsSum
	expectedB0 := 0.0 - 0.1*(prob0-1.0)
	expectedB1 := 0.0 - 0.1*(prob1-0.0)

	t.Logf("updated W_out=%v B_out=%v", net.W_out, net.B_out)
	if diff := math.Abs(net.W_out[0][0] - expectedWOut0); diff > 1e-9 {
		t.Fatalf("unexpected W_out[0][0]: diff=%g", diff)
	}
	if diff := math.Abs(net.W_out[0][1] - expectedWOut1); diff > 1e-9 {
		t.Fatalf("unexpected W_out[0][1]: diff=%g", diff)
	}
	if diff := math.Abs(net.B_out[0] - expectedB0); diff > 1e-9 {
		t.Fatalf("unexpected B_out[0]: diff=%g", diff)
	}
	if diff := math.Abs(net.B_out[1] - expectedB1); diff > 1e-9 {
		t.Fatalf("unexpected B_out[1]: diff=%g", diff)
	}
}
