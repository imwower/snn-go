package data_test

import (
	"strings"
	"testing"

	"github.com/imwower/snn-go/internal/data"
)

func TestNewLoaderMNISTMissingData(t *testing.T) {
	t.Helper()
	const (
		root      = "./testdata/non-existent"
		batchSize = 32
		seed      = 42
	)
	loader, err := data.NewLoaderMNIST(root, batchSize, seed)
	if err == nil {
		t.Fatalf("expected error for missing dataset")
	}
	if loader != nil {
		t.Fatalf("expected nil loader on error")
	}
	if !strings.Contains(err.Error(), "read training images") {
		t.Fatalf("unexpected error message: %v", err)
	}
}
