package zarr

import (
	"strconv"
	"strings"
)

// GridShape calculates the number of chunks in each dimension.
// For each dimension i, the number of chunks is ceil(shape[i] / chunks[i]).
func GridShape(shape, chunks []int) []int {
	if len(shape) == 0 || len(chunks) == 0 {
		return []int{} // 0D scalar
	}
	grid := make([]int, len(shape))
	for i := range shape {
		grid[i] = (shape[i] + chunks[i] - 1) / chunks[i]
	}
	return grid
}

// ChunkKey generates the key for a chunk given its indices and a separator.
// For Zarr V2, the separator is typically ".".
// Example: indices=[1, 4], separator="." -> "1.4"
// For 0D arrays (empty indices), it returns "0" per the Zarr spec.
func ChunkKey(indices []int, separator string) string {
	if len(indices) == 0 {
		return "0"
	}

	if len(indices) == 1 {
		return strconv.Itoa(indices[0])
	}

	var sb strings.Builder
	for i, idx := range indices {
		if i > 0 {
			sb.WriteString(separator)
		}
		sb.WriteString(strconv.Itoa(idx))
	}
	return sb.String()
}
