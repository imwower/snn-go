package data_test

import (
	"testing"

	"github.com/imwower/snn-go/internal/data"
)

func TestNewLoaderMNISTFallbackSynth(t *testing.T) {
	t.Helper()
	const (
		root      = "./testdata/non-existent"
		batchSize = 32
		seed      = 42
	)
	loader, err := data.NewLoaderMNIST(root, batchSize, seed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	first, ok := loader.Next()
	if !ok {
		t.Fatalf("expected first batch, got none")
	}
	if len(first.X) != batchSize {
		t.Fatalf("unexpected first batch size: got %d want %d", len(first.X), batchSize)
	}
	if len(first.Y) != batchSize {
		t.Fatalf("unexpected first labels len: got %d want %d", len(first.Y), batchSize)
	}
	for i, row := range first.X {
		if len(row) != 784 {
			t.Fatalf("row %d has unexpected length %d", i, len(row))
		}
		for j, v := range row {
			if v < 0 || v > 1 {
				t.Fatalf("pixel out of range at [%d][%d]: %f", i, j, v)
			}
		}
	}
	for i, label := range first.Y {
		if label < 0 || label > 9 {
			t.Fatalf("label out of range at index %d: %d", i, label)
		}
	}
	total := len(first.X)
	batches := 1
	for {
		batch, ok := loader.Next()
		if !ok {
			break
		}
		total += len(batch.X)
		batches++
	}
	t.Logf("drained %d samples across %d batches from synthetic loader", total, batches)
	if total != 10000 {
		t.Fatalf("unexpected total sample count: got %d want %d", total, 10000)
	}
}
