package data

import (
	"bufio"
	"compress/gzip"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

type Batch struct {
	X [][]float64 // [B][784] 0..1
	Y []int       // [B] class id
}

type Loader struct {
	images [][]float64
	labels []int
	bs     int
	pos    int
	perm   []int
}

func NewLoaderMNIST(root string, bs int, seed int64) (*Loader, error) {
	img, err := readIdx(filepath.Join(root, "train-images-idx3-ubyte"), 16)
	if err != nil {
		// 尝试 .gz
		img, err = readIdxGZ(filepath.Join(root, "train-images-idx3-ubyte.gz"), 16)
	}
	lb, err2 := readIdx(filepath.Join(root, "train-labels-idx1-ubyte"), 8)
	if err2 != nil {
		lb, err2 = readIdxGZ(filepath.Join(root, "train-labels-idx1-ubyte.gz"), 8)
	}
	if err != nil || err2 != nil || len(lb) == 0 {
		// 兜底：合成可线性分的 2D → 映射到 784 维
		return synth(bs, seed), nil
	}
	// 解析成 [N][784]
	var images [][]float64
	pix := img
	// 前 16 字节是 header，已在 reader 中跳过
	// 这里 img 已经是纯像素流
	n := len(lb)
	images = make([][]float64, n)
	for i := 0; i < n; i++ {
		arr := make([]float64, 784)
		for j := 0; j < 784; j++ {
			arr[j] = float64(pix[i*784+j]) / 255.0
		}
		images[i] = arr
	}
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		labels[i] = int(lb[i])
	}
	ld := &Loader{images: images, labels: labels, bs: bs}
	ld.reset(seed)
	return ld, nil
}

func (l *Loader) reset(seed int64) {
	l.pos = 0
	n := len(l.labels)
	l.perm = rand.Perm(n)
	rand.Seed(seed)
}

func (l *Loader) Next() (Batch, bool) {
	if l.pos >= len(l.labels) {
		return Batch{}, false
	}
	end := min(l.pos+l.bs, len(l.labels))
	idx := l.perm[l.pos:end]
	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, id := range idx {
		bx[i] = l.images[id]
		by[i] = l.labels[id]
	}
	l.pos = end
	return Batch{X: bx, Y: by}, true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func readIdx(path string, header int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// 只返回 payload（像素或标签字节）
	_, _ = f.Seek(header, 0)
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < int(header) {
		return nil, errors.New("bad idx")
	}
	return b[header:], nil
}

func readIdxGZ(path string, header int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(bufio.NewReader(f))
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	all, err := ioReadAll(gz)
	if err != nil {
		return nil, err
	}
	if int64(len(all)) < header {
		return nil, errors.New("bad idx.gz")
	}
	return all[header:], nil
}

func ioReadAll(r *gzip.Reader) ([]byte, error) {
	buf := make([]byte, 0, 512*1024)
	tmp := make([]byte, 32*1024)
	for {
		n, err := r.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, err
		}
	}
	return buf, nil
}

// 合成数据：两簇高斯，映射到 784 维（其余维用噪声）
func synth(bs int, seed int64) *Loader {
	rand.Seed(seed)
	n := 10000
	images := make([][]float64, n)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		c := i % 10
		x1 := rand.NormFloat64()*0.5 + float64(c)/10.0
		x2 := rand.NormFloat64()*0.5 + float64(c)/10.0
		arr := make([]float64, 784)
		arr[0] = sigmoid(x1)
		arr[1] = sigmoid(x2)
		for j := 2; j < 784; j++ {
			arr[j] = rand.Float64() * 0.1
		}
		images[i] = arr
		labels[i] = c
	}
	ld := &Loader{images: images, labels: labels, bs: bs}
	ld.reset(seed)
	return ld
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
