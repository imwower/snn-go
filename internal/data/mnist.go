package data

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

type Batch struct {
	X [][]float64 // [B][784] 归一化像素 0..1
	Y []int       // [B] 类别编号
}

type Loader struct {
	images [][]float64
	labels []int
	bs     int
	pos    int
	perm   []int
	seed   int64
}

func NewLoaderMNIST(root string, bs int, seed int64) (*Loader, error) {
	img, err := readIdx(filepath.Join(root, "train-images-idx3-ubyte"), 16)
	if err != nil {
		img, _ = readIdxGZ(filepath.Join(root, "train-images-idx3-ubyte.gz"), 16)
	}
	lb, err2 := readIdx(filepath.Join(root, "train-labels-idx1-ubyte"), 8)
	if err2 != nil {
		lb, _ = readIdxGZ(filepath.Join(root, "train-labels-idx1-ubyte.gz"), 8)
	}
	// 若失败则回退到合成数据
	if len(img) == 0 || len(lb) == 0 {
		log.Printf("mnist：%s 缺少数据集，回退到合成数据", root)
		return synth(bs, seed), nil
	}

	n := len(lb)
	if len(img) < n*784 {
		return nil, errors.New("image length < labels*784")
	}
	images := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, 784)
		for j := 0; j < 784; j++ {
			row[j] = float64(img[i*784+j]) / 255.0
		}
		images[i] = row
	}
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		labels[i] = int(lb[i])
	}
	log.Printf("mnist：已从 %s 载入 %d 条样本", root, n)
	ld := &Loader{images: images, labels: labels, bs: bs, seed: seed}
	ld.reset()
	return ld, nil
}

func (l *Loader) reset() {
	l.pos = 0
	n := len(l.labels)
	r := rand.New(rand.NewSource(l.seed))
	l.perm = r.Perm(n)
}

func (l *Loader) Next() (Batch, bool) {
	if l.pos >= len(l.labels) {
		return Batch{}, false
	}
	end := l.pos + l.bs
	if end > len(l.labels) {
		end = len(l.labels)
	}
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

func readIdx(path string, header int) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < header {
		return nil, errors.New("bad idx header")
	}
	return b[header:], nil
}

func readIdxGZ(path string, header int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	all, err := io.ReadAll(gz)
	if err != nil {
		return nil, err
	}
	if len(all) < header {
		return nil, errors.New("bad idx.gz header")
	}
	return all[header:], nil
}

// —— 合成数据兜底：10 类，二维可分，高维噪声 —— //

func synth(bs int, seed int64) *Loader {
	r := rand.New(rand.NewSource(seed))
	n := 10000
	images := make([][]float64, n)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		c := i % 10
		x1 := r.NormFloat64()*0.5 + float64(c)/10.0
		x2 := r.NormFloat64()*0.5 + float64(c)/10.0
		arr := make([]float64, 784)
		arr[0] = sigmoid(x1)
		arr[1] = sigmoid(x2)
		for j := 2; j < 784; j++ {
			arr[j] = r.Float64() * 0.1
		}
		images[i] = arr
		labels[i] = c
	}
	ld := &Loader{images: images, labels: labels, bs: bs, seed: seed}
	ld.reset()
	return ld
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// （可选）简单二进制读工具：未使用，但保留做 idx 兼容
func readBigEndianInt32(b []byte) int {
	return int(binary.BigEndian.Uint32(b))
}
