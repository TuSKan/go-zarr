package zarr_test

import (
	"reflect"
	"testing"

	"github.com/TuSKan/go-zarr/zarr"
)

func TestGridShape(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		chunks   []int
		expected []int
	}{
		{
			name:     "Shape [128, 128], Chunks [64, 64]",
			shape:    []int{128, 128},
			chunks:   []int{64, 64},
			expected: []int{2, 2},
		},
		{
			name:     "Shape [100], Chunks [30]",
			shape:    []int{100},
			chunks:   []int{30},
			expected: []int{4},
		},
		{
			name:     "Shape [], Chunks []",
			shape:    []int{},
			chunks:   []int{},
			expected: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := zarr.GridShape(tt.shape, tt.chunks)
			if !reflect.DeepEqual(got, tt.expected) {
				// Handle both being empty correctly, since DeepEqual handles nil vs empty differently sometimes.
				if len(got) == 0 && len(tt.expected) == 0 {
					return
				}
				t.Errorf("GridShape(%v, %v) = %v, want %v", tt.shape, tt.chunks, got, tt.expected)
			}
		})
	}
}

func TestChunkKey(t *testing.T) {
	tests := []struct {
		name     string
		indices  []int
		sep      string
		expected string
	}{
		{
			name:     "Indices [1, 0], sep .",
			indices:  []int{1, 0},
			sep:      ".",
			expected: "1.0",
		},
		{
			name:     "Indices [0, 2, 5], sep /",
			indices:  []int{0, 2, 5},
			sep:      "/",
			expected: "0/2/5",
		},
		{
			name:     "Indices [], sep .",
			indices:  []int{},
			sep:      ".",
			expected: "0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := zarr.ChunkKey(tt.indices, tt.sep)
			if got != tt.expected {
				t.Errorf("ChunkKey(%v, %q) = %q, want %q", tt.indices, tt.sep, got, tt.expected)
			}
		})
	}
}
