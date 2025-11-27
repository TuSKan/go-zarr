package zarr

import "testing"

func TestChunkKey(t *testing.T) {
	tests := []struct {
		indices   []int
		separator string
		expected  string
	}{
		{[]int{1, 4}, ".", "1.4"},
		{[]int{0, 0, 0}, ".", "0.0.0"},
		{[]int{10}, ".", "10"},
		{[]int{1, 2}, "/", "1/2"}, // Test different separator
		// {[]int{}, ".", "0"}, // TODO: Verify behavior for 0-d arrays if needed
	}

	for _, tt := range tests {
		got := ChunkKey(tt.indices, tt.separator)
		if got != tt.expected {
			t.Errorf("ChunkKey(%v, %q) = %q, want %q", tt.indices, tt.separator, got, tt.expected)
		}
	}
}
