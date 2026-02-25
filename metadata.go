package zarr

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// CompressorConfig represents the Zarr compressor metadata.
type CompressorConfig struct {
	ID      string `json:"id"`
	Cname   string `json:"cname,omitempty"`
	Clevel  int    `json:"clevel,omitempty"`
	Shuffle int    `json:"shuffle,omitempty"`
}

// Metadata represents the Zarr V2 .zarray metadata.
type Metadata struct {
	ZarrFormat int               `json:"zarr_format"`
	Shape      []int             `json:"shape"`
	Chunks     []int             `json:"chunks"`
	DType      string            `json:"dtype"`
	Compressor *CompressorConfig `json:"compressor"`
	FillValue  interface{}       `json:"fill_value"`
	Order      string            `json:"order"`
}

// LoadMetadata reads and parses the .zarray file from the given directory path.
func LoadMetadata(reader io.Reader) (*Metadata, error) {
	var meta Metadata
	if err := json.NewDecoder(reader).Decode(&meta); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	if meta.ZarrFormat != 2 {
		return nil, fmt.Errorf("unsupported zarr_format: %d, expected 2", meta.ZarrFormat)
	}

	return &meta, nil
}

// ParseDType takes a numpy-style string like "<f4", "|b1", "<i8",
// and returns a simplified string name (e.g., "float32", "bool", "int64"),
// the byte size (e.g., 4, 1, 8), and an error if unsupported.
// Reject big-endian (>) types for now.
func ParseDType(s string) (string, int, error) {
	if len(s) < 3 {
		return "", 0, fmt.Errorf("invalid dtype: %s", s)
	}

	endian := s[0]
	if endian == '>' {
		return "", 0, fmt.Errorf("big-endian types are unsupported: %s", s)
	}

	kind := s[1]
	sizeStr := s[2:]

	size, err := strconv.Atoi(sizeStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid size in dtype: %s", s)
	}

	switch kind {
	case 'b':
		return "bool", size, nil
	case 'i':
		return fmt.Sprintf("int%d", size*8), size, nil
	case 'u':
		return fmt.Sprintf("uint%d", size*8), size, nil
	case 'f':
		return fmt.Sprintf("float%d", size*8), size, nil
	case 'c':
		return fmt.Sprintf("complex%d", size*8), size, nil
	default:
		return "", 0, fmt.Errorf("unsupported dtype kind: %c in %s", kind, s)
	}
}
