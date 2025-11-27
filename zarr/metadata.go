package zarr

import (
	"fmt"
)

// Metadata represents the Zarr V2 .zarray metadata.
// Metadata represents the Zarr V2 .zarray metadata.
type Metadata struct {
	Chunks     []int             `json:"chunks"`
	Compressor *CompressorConfig `json:"compressor"`
	DType      string            `json:"dtype"`
	Shape      []int             `json:"shape"`
	ZarrFormat int               `json:"zarr_format"`
}

// CompressorConfig represents the compression configuration.
type CompressorConfig struct {
	ID    string `json:"id"`
	Level int    `json:"level,omitempty"`
}

// DType represents a parsed Zarr data type.
type DType string

// Common Zarr DTypes.
const (
	Bool    DType = "bool"
	Int8    DType = "int8"
	Int16   DType = "int16"
	Int32   DType = "int32"
	Int64   DType = "int64"
	Uint8   DType = "uint8"
	Uint16  DType = "uint16"
	Uint32  DType = "uint32"
	Uint64  DType = "uint64"
	Float32 DType = "float32"
	Float64 DType = "float64"
	// Complex types can be added if needed
)

// ParseDType parses a Zarr dtype string (e.g., "<f4", "|b1") into a Go-friendly DType.
func ParseDType(dtype string) (DType, error) {
	if len(dtype) < 2 {
		return "", fmt.Errorf("invalid dtype: %s", dtype)
	}

	// Handle simple cases or numpy-style strings
	// Zarr spec: https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding
	// Basic mapping for common types:
	switch dtype {
	case "|b1":
		return Bool, nil
	case "|i1":
		return Int8, nil
	case "|u1":
		return Uint8, nil
	case "<i2":
		return Int16, nil
	case "<i4":
		return Int32, nil
	case "<i8":
		return Int64, nil
	case "<u2":
		return Uint16, nil
	case "<u4":
		return Uint32, nil
	case "<u8":
		return Uint64, nil
	case "<f4":
		return Float32, nil
	case "<f8":
		return Float64, nil
	// Big-endian variants (>) could be added if needed, but assuming little-endian (<) for now as it's standard on x86
	case ">i2":
		return Int16, nil // Note: This doesn't handle endianness conversion, just type mapping
	case ">i4":
		return Int32, nil
	case ">i8":
		return Int64, nil
	case ">u2":
		return Uint16, nil
	case ">u4":
		return Uint32, nil
	case ">u8":
		return Uint64, nil
	case ">f4":
		return Float32, nil
	case ">f8":
		return Float64, nil
	}

	return "", fmt.Errorf("unsupported or unknown dtype: %s", dtype)
}
