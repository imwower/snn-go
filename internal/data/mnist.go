package data

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"log"
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
	imgPath := filepath.Join(root, "train-images-idx3-ubyte")
	imgGZPath := imgPath + ".gz"
	img, err := readIdx(imgPath, 16)
	if err != nil {
		img, err = readIdxGZ(imgGZPath, 16)
		if err != nil {
			return nil, fmt.Errorf("mnist: read training images: %w", err)
		}
	}

	labelPath := filepath.Join(root, "train-labels-idx1-ubyte")
	labelGZPath := labelPath + ".gz"
	lb, err := readIdx(labelPath, 8)
	if err != nil {
		lb, err = readIdxGZ(labelGZPath, 8)
		if err != nil {
			return nil, fmt.Errorf("mnist: read training labels: %w", err)
		}
	}

	if len(img) == 0 || len(lb) == 0 {
		return nil, fmt.Errorf("mnist: dataset under %s is empty", root)
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
