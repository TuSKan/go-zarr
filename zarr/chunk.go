package zarr

import (
	"strconv"
	"strings"
)

// ChunkKey generates the key for a chunk given its indices and a separator.
// For Zarr V2, the separator is typically ".".
// Example: indices=[1, 4], separator="." -> "1.4"
func ChunkKey(indices []int, separator string) string {
	if len(indices) == 0 {
		return "0" // Zarr spec for 0-d arrays or just convention? Actually spec says "0" for 0-d.
		// But for empty indices slice?
		// If shape is empty (scalar), indices is empty. Key is "0".
		// Let's assume standard case first.
	}

	// Optimization for common case
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
