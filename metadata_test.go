package zarr_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/TuSKan/go-zarr"
)

func TestParseDType(t *testing.T) {
	tests := []struct {
		input       string
		expectedStr string
		expectedSz  int
		expectErr   bool
	}{
		{"<f4", "float32", 4, false},
		{"<i8", "int64", 8, false},
		{"|b1", "bool", 1, false},
		{">f4", "", 0, true}, // big-endian should fail
		{"x2", "", 0, true},  // invalid encoding
		{"<x4", "", 0, true}, // unknown kind
		{"<i", "", 0, true},  // incomplete size
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			str, sz, err := zarr.ParseDType(tt.input)

			if tt.expectErr {
				if err == nil {
					t.Errorf("expected error for input %q, but got nil", tt.input)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error for input %q: %v", tt.input, err)
				}
				if str != tt.expectedStr {
					t.Errorf("expected string %q, got %q", tt.expectedStr, str)
				}
				if sz != tt.expectedSz {
					t.Errorf("expected size %d, got %d", tt.expectedSz, sz)
				}
			}
		})
	}
}

func TestLoadMetadata(t *testing.T) {
	tempDir := t.TempDir()

	mockJSON := `{
		"zarr_format": 2,
		"shape": [128, 128],
		"chunks": [64, 64],
		"dtype": "<f4",
		"compressor": null,
		"fill_value": 0.0,
		"order": "C"
	}`

	zarrayPath := filepath.Join(tempDir, ".zarray")
	if err := os.WriteFile(zarrayPath, []byte(mockJSON), 0644); err != nil {
		t.Fatalf("failed to write mock json: %v", err)
	}

	f, err := os.Open(zarrayPath)
	if err != nil {
		t.Fatalf("failed to open mock json: %v", err)
	}
	defer f.Close()

	meta, err := zarr.LoadMetadata(f)
	if err != nil {
		t.Fatalf("LoadMetadata failed: %v", err)
	}

	expectedShape := []int{128, 128}
	if !reflect.DeepEqual(meta.Shape, expectedShape) {
		t.Errorf("expected shape %v, got %v", expectedShape, meta.Shape)
	}

	expectedChunks := []int{64, 64}
	if !reflect.DeepEqual(meta.Chunks, expectedChunks) {
		t.Errorf("expected chunks %v, got %v", expectedChunks, meta.Chunks)
	}

	if meta.ZarrFormat != 2 {
		t.Errorf("expected zarr_format 2, got %d", meta.ZarrFormat)
	}

	if meta.DType != "<f4" {
		t.Errorf("expected dtype <f4, got %s", meta.DType)
	}
}
