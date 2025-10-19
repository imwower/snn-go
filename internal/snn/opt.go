package snn

import (
	"runtime"
	"sync"
)

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func gemvVec(v []float64, wt [][]float64, out []float64) {
	for h := 0; h < len(wt); h++ {
		out[h] += dot(v, wt[h])
	}
}

func gemvVecParallel(v []float64, wt [][]float64, out []float64, workers int) {
	if workers <= 1 || len(wt) < 64 {
		gemvVec(v, wt, out)
		return
	}
	n := len(wt)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if start >= n {
			wg.Done()
			continue
		}
		if end > n {
			end = n
		}
		go func(s, e int) {
			defer wg.Done()
			for h := s; h < e; h++ {
				out[h] += dot(v, wt[h])
			}
		}(start, end)
	}
	wg.Wait()
}

func defaultWorkers() int {
	w := runtime.GOMAXPROCS(0)
	if w > 4 {
		return 4
	}
	if w < 1 {
		return 1
	}
	return w
}
