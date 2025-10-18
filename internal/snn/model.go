package snn

import (
	"math"
	"math/rand"
)

type ThreeCompNet struct {
	Input, Hidden, Output int

	// 权重：输入→基底；基底/顶→soma；读出
	W_in  [][]float64 // [input][hidden]
	W_b2s [][]float64 // [hidden][hidden]
	W_a2s [][]float64 // [hidden][hidden]
	W_out [][]float64 // [hidden][output]
	B_out []float64

	// 超参
	Theta                  float64
	AlphaB, AlphaA, AlphaS float64
	KappaBS, KappaAS       float64

	// 状态（训练时分配）
}

func NewThreeCompNet(in, h, out int, seed int64, theta, ab, aa, as, kbs, kas float64) *ThreeCompNet {
	rand.Seed(seed)
	m := &ThreeCompNet{
		Input: in, Hidden: h, Output: out,
		Theta: theta, AlphaB: ab, AlphaA: aa, AlphaS: as, KappaBS: kbs, KappaAS: kas,
		W_in:  randMat(in, h, 0.05),
		W_b2s: randMat(h, h, 0.02),
		W_a2s: randMat(h, h, 0.02),
		W_out: randMat(h, out, 0.1),
		B_out: make([]float64, out),
	}
	return m
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

// 前向：固定 T 个时间步，返回聚合 logits 与中间缓存（用于梯度）
type ForwardCache struct {
	Vb [][]float64 // [T][H]
	Va [][]float64
	Vs [][]float64
	S  [][]float64
}

func (m *ThreeCompNet) Forward(x [][]float64, T int) (logits [][]float64, cache ForwardCache) {
	// x: [B][Input], 作为恒定率，时间展开时每步都喂同一输入（也可以做 Poisson 发放）
	B := len(x)
	H, O := m.Hidden, m.Output
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
	vs := make([]float64, B*H) // 当前 soma
	vb := make([]float64, B*H)
	va := make([]float64, B*H)

	agg := make([][]float64, B) // 聚合到输出的 soma 电压
	for i := 0; i < B; i++ {
		agg[i] = make([]float64, O)
	}

	for t := 0; t < T; t++ {
		// basal 更新
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				// vb_{t+1} = (1-ab)*vb + x·W_in + kbs*(vs - vb)
				idx := b*H + h
				sumIn := 0.0
				for d := 0; d < m.Input; d++ {
					sumIn += x[b][d] * m.W_in[d][h]
				}
				vb[idx] = (1-m.AlphaB)*vb[idx] + sumIn + m.KappaBS*(vs[idx]-vb[idx])
			}
		}
		// apical 更新（这里不接外部反馈，保留与 soma 的耦合）
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				idx := b*H + h
				va[idx] = (1-m.AlphaA)*va[idx] + m.KappaAS*(vs[idx]-va[idx])
			}
		}
		// soma 电位 + 发放
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				idx := b*H + h
				z := (1 - m.AlphaS) * vs[idx]
				// vb→soma
				for k := 0; k < H; k++ {
					z += vb[b*H+k] * m.W_b2s[k][h]
				}
				// va→soma
				for k := 0; k < H; k++ {
					z += va[b*H+k] * m.W_a2s[k][h]
				}
				// spike
				sp := float64(heaviside(z - m.Theta))
				vs[idx] = z - m.Theta*sp

				cache.Vb[t][idx] = vb[idx]
				cache.Va[t][idx] = va[idx]
				cache.Vs[t][idx] = vs[idx]
				cache.S[t][idx] = sp
			}
		}
		// 读出（聚合 soma）
		for b := 0; b < B; b++ {
			for o := 0; o < O; o++ {
				// 简化读出：Σ(h) vs[b,h] * W_out[h,o]
				for h := 0; h < H; h++ {
					agg[b][o] += vs[b*H+h] * m.W_out[h][o]
				}
			}
		}
	}
	// logits = agg + B_out
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
	// 数值稳定
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

// 只对 W_out 做 CE 梯度（读出层可训练，骨干做无梯度近似/可选 STE）
// 这样能稳定提高 acc，且实现简洁；若要端到端，可增加对 W_in/W_b2s 的近似梯度（略）。
func (m *ThreeCompNet) BackpropReadout(cache ForwardCache, x [][]float64, logits [][]float64, y []int, lr float64) (batchLoss float64, batchAcc float64) {
	B := len(x)
	H := m.Hidden
	O := m.Output
	// 反向：dL/dW_out = sum_b ( (p - y_onehot) ⊗ Σ_t vs_t )
	// 我们已经把 Σ_t vs_t*W_out 聚到了 logits；此处需要 Σ_t vs_t（可由 cache 累加）
	vsAgg := make([][]float64, B) // [B][H]
	for b := 0; b < B; b++ {
		vsAgg[b] = make([]float64, H)
		for t := 0; t < len(cache.Vs); t++ {
			for h := 0; h < H; h++ {
				vsAgg[b][h] += cache.Vs[t][b*H+h]
			}
		}
	}
	// 逐样本计算 CE
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

		// 梯度更新读出层
		for o := 0; o < O; o++ {
			g := probs[o] - bool01(y[b] == o) // dL/dz
			for h := 0; h < H; h++ {
				m.W_out[h][o] -= lr * g * vsAgg[b][h]
			}
		}
		for o := 0; o < O; o++ {
			m.B_out[o] -= lr * (probs[o] - bool01(y[b] == o))
		}
	}
	return batchLoss / float64(B), batchAcc / float64(B)
}

func bool01(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
