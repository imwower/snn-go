package snn

import (
	"math"
	"math/rand"
)

type ThreeCompNet struct {
	Input, Hidden, Output int

	W_in  [][]float64 // [input][hidden]
	W_b2s [][]float64 // [hidden][hidden]
	W_a2s [][]float64 // [hidden][hidden]
	W_out [][]float64 // [hidden][output]
	B_out []float64

	Theta                  float64
	AlphaB, AlphaA, AlphaS float64
	KappaBS, KappaAS       float64
}

func NewThreeCompNet(in, h, out int, seed int64, theta, ab, aa, as, kbs, kas float64) *ThreeCompNet {
	rand.Seed(seed)
	return &ThreeCompNet{
		Input: in, Hidden: h, Output: out,
		W_in:  randMat(in, h, 0.05),
		W_b2s: randMat(h, h, 0.02),
		W_a2s: randMat(h, h, 0.02),
		W_out: randMat(h, out, 0.10),
		B_out: make([]float64, out),
		Theta: theta, AlphaB: ab, AlphaA: aa, AlphaS: as, KappaBS: kbs, KappaAS: kas,
	}
}

func randMat(r, c int, scale float64) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			m[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}
	return m
}

type ForwardCache struct {
	Vb [][]float64 // [T][B*H]
	Va [][]float64
	Vs [][]float64
	S  [][]float64
}

func (m *ThreeCompNet) Forward(x [][]float64, T int) (logits [][]float64, cache ForwardCache) {
	B, H, O := len(x), m.Hidden, m.Output
	cache.Vb = make([][]float64, T)
	cache.Va = make([][]float64, T)
	cache.Vs = make([][]float64, T)
	cache.S = make([][]float64, T)
	for t := 0; t < T; t++ {
		cache.Vb[t] = make([]float64, B*H)
		cache.Va[t] = make([]float64, B*H)
		cache.Vs[t] = make([]float64, B*H)
		cache.S[t] = make([]float64, B*H)
	}

	vb := make([]float64, B*H)
	va := make([]float64, B*H)
	vs := make([]float64, B*H)

	agg := make([][]float64, B) // Σ_t vs · W_out
	for b := 0; b < B; b++ {
		agg[b] = make([]float64, O)
	}

	for t := 0; t < T; t++ {
		// basal
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				idx := b*H + h
				sumIn := 0.0
				for d := 0; d < m.Input; d++ {
					sumIn += x[b][d] * m.W_in[d][h]
				}
				vb[idx] = (1-m.AlphaB)*vb[idx] + sumIn + m.KappaBS*(vs[idx]-vb[idx])
			}
		}
		// apical
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				idx := b*H + h
				va[idx] = (1-m.AlphaA)*va[idx] + m.KappaAS*(vs[idx]-va[idx])
			}
		}
		// soma + spike
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				idx := b*H + h
				z := (1 - m.AlphaS) * vs[idx]
				for k := 0; k < H; k++ {
					z += vb[b*H+k]*m.W_b2s[k][h] + va[b*H+k]*m.W_a2s[k][h]
				}
				sp := heaviside(z - m.Theta)
				vs[idx] = z - m.Theta*float64(sp)

				cache.Vb[t][idx] = vb[idx]
				cache.Va[t][idx] = va[idx]
				cache.Vs[t][idx] = vs[idx]
				cache.S[t][idx] = float64(sp)
			}
		}
		// 读出聚合
		for b := 0; b < B; b++ {
			for o := 0; o < O; o++ {
				sum := 0.0
				for h := 0; h < H; h++ {
					sum += vs[b*H+h] * m.W_out[h][o]
				}
				agg[b][o] += sum
			}
		}
	}
	// logits = Σ_t vs·W_out + b
	logits = make([][]float64, B)
	for b := 0; b < B; b++ {
		logits[b] = make([]float64, O)
		for o := 0; o < O; o++ {
			logits[b][o] = agg[b][o] + m.B_out[o]
		}
	}
	return logits, cache
}

func heaviside(x float64) int {
	if x > 0 {
		return 1
	}
	return 0
}

func softmaxCE(logits []float64, y int) (loss float64, probs []float64) {
	maxv := -1e30
	for _, v := range logits {
		if v > maxv {
			maxv = v
		}
	}
	sum := 0.0
	probs = make([]float64, len(logits))
	for i, v := range logits {
		p := math.Exp(v - maxv)
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}
	loss = -math.Log(probs[y] + 1e-12)
	return
}

func (m *ThreeCompNet) BackpropReadout(cache ForwardCache, x [][]float64, logits [][]float64, y []int, lr float64) (float64, float64) {
	B := len(x)
	H := m.Hidden
	O := m.Output
	vsAgg := make([][]float64, B)
	for b := 0; b < B; b++ {
		vsAgg[b] = make([]float64, H)
		for t := 0; t < len(cache.Vs); t++ {
			for h := 0; h < H; h++ {
				vsAgg[b][h] += cache.Vs[t][b*H+h]
			}
		}
	}
	var batchLoss, batchAcc float64
	for b := 0; b < B; b++ {
		loss, probs := softmaxCE(logits[b], y[b])
		batchLoss += loss
		// acc
		arg := 0
		for o := 1; o < O; o++ {
			if probs[o] > probs[arg] {
				arg = o
			}
		}
		if arg == y[b] {
			batchAcc += 1.0
		}
		// dL/dW_out = (p - y_onehot) ⊗ Σ_t vs
		for o := 0; o < O; o++ {
			g := probs[o] - b2f(y[b] == o)
			for h := 0; h < H; h++ {
				m.W_out[h][o] -= lr * g * vsAgg[b][h]
			}
			m.B_out[o] -= lr * g
		}
	}
	return batchLoss / float64(B), batchAcc / float64(B)
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
