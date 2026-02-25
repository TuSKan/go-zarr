package zarr

import (
	"bytes"
	"compress/zlib"
	"context"
	"fmt"
	"io"

	"github.com/mrjoshuak/go-blosc"
	"gocloud.dev/blob"
	"gocloud.dev/gcerrors"
)

type Reader struct {
	bucket *blob.Bucket
	meta   *Metadata
}

func NewReader(ctx context.Context, path string) (*Reader, error) {
	bucket, err := blob.OpenBucket(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to create bucket: %w", err)
	}

	reader, err := bucket.NewReader(ctx, ".zarray", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open .zarray: %w", err)
	}
	defer reader.Close()

	meta, err := LoadMetadata(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}
	return &Reader{
		bucket: bucket,
		meta:   meta,
	}, nil
}

// strides computes the C-order strides for a given shape.
func strides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	s := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = stride
		stride *= shape[i]
	}
	return s
}

// ReadFull reads the entire Zarr array into a flat byte slice.
func (r *Reader) ReadFull(ctx context.Context) ([]byte, error) {
	// Parse dtype to get item size
	_, itemSize, err := ParseDType(r.meta.DType)
	if err != nil {
		return nil, fmt.Errorf("invalid dtype: %w", err)
	}

	totalElements := 1
	for _, dim := range r.meta.Shape {
		totalElements *= dim
	}

	totalBytes := totalElements * itemSize
	buffer := make([]byte, totalBytes)

	// If 0D, read the single chunk "0" and return
	if len(r.meta.Shape) == 0 {
		reader, err := r.bucket.NewReader(ctx, "0", nil)
		if err != nil {
			if gcerrors.Code(err) == gcerrors.NotFound {
				return buffer, nil // fill value (0)
			}
			return nil, fmt.Errorf("failed to open 0D chunk: %w", err)
		}
		defer reader.Close()
		_, err = io.ReadFull(reader, buffer)
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
			return nil, fmt.Errorf("failed to read 0D chunk: %w", err)
		}
		return buffer, nil
	}

	grid := GridShape(r.meta.Shape, r.meta.Chunks)
	globalStrides := strides(r.meta.Shape)
	chunkStrides := strides(r.meta.Chunks)

	// Recursive function to iterate over all chunk coordinates
	var iterateChunks func(dim int, currentCoords []int) error
	iterateChunks = func(dim int, currentCoords []int) error {
		if dim == len(grid) {
			return r.processChunk(ctx, currentCoords, buffer, itemSize, globalStrides, chunkStrides)
		}

		for i := 0; i < grid[dim]; i++ {
			currentCoords[dim] = i
			if err := iterateChunks(dim+1, currentCoords); err != nil {
				return err
			}
		}
		return nil
	}

	coords := make([]int, len(grid))
	if err := iterateChunks(0, coords); err != nil {
		return nil, err
	}

	return buffer, nil
}

// ReadChunk reads a single chunk from the Zarr array given its coordinates.
func (r *Reader) ReadChunk(ctx context.Context, coords []int) ([]byte, error) {
	key := ChunkKey(coords, ".")

	reader, err := r.bucket.NewReader(ctx, key, nil)
	if err != nil {
		if gcerrors.Code(err) == gcerrors.NotFound {
			// Chunk missing, calculate expected size and return zero-filled array
			_, itemSize, err := ParseDType(r.meta.DType)
			if err != nil {
				return nil, fmt.Errorf("invalid dtype: %w", err)
			}
			expectedElements := 1
			for _, dim := range r.meta.Chunks {
				expectedElements *= dim
			}
			return make([]byte, expectedElements*itemSize), nil
		}
		return nil, fmt.Errorf("failed to open chunk %s: %w", key, err)
	}
	defer reader.Close()

	chunkData, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read chunk %s: %w", key, err)
	}

	if r.meta.Compressor != nil {
		switch r.meta.Compressor.ID {
		case "blosc":
			chunkData, err = blosc.Decompress(chunkData)
			if err != nil {
				return nil, fmt.Errorf("failed to decompress blosc chunk %s: %w", key, err)
			}
		case "zlib", "gzip":
			zr, err := zlib.NewReader(bytes.NewReader(chunkData))
			if err != nil {
				return nil, fmt.Errorf("failed to init zlib reader for chunk %s: %w", key, err)
			}
			chunkData, err = io.ReadAll(zr)
			zr.Close()
			if err != nil {
				return nil, fmt.Errorf("failed to decompress zlib chunk %s: %w", key, err)
			}
		default:
			return nil, fmt.Errorf("unsupported compressor: %s", r.meta.Compressor.ID)
		}
	}

	return chunkData, nil
}

func (r *Reader) processChunk(ctx context.Context, chunkCoords []int, globalBuffer []byte, itemSize int, globalStrides, chunkStrides []int) error {
	chunkData, err := r.ReadChunk(ctx, chunkCoords)
	if err != nil {
		return err
	}

	// Calculate bounds for this chunk within the global array
	chunkStartGlobal := make([]int, len(r.meta.Shape))
	chunkShape := make([]int, len(r.meta.Shape))
	for i, coord := range chunkCoords {
		chunkStartGlobal[i] = coord * r.meta.Chunks[i]
		endGlobal := chunkStartGlobal[i] + r.meta.Chunks[i]
		if endGlobal > r.meta.Shape[i] {
			endGlobal = r.meta.Shape[i]
		}
		chunkShape[i] = endGlobal - chunkStartGlobal[i]
	}

	// Recursive function to iterate over elements within the chunk
	var copyElements func(dim int, relCoords []int)
	copyElements = func(dim int, relCoords []int) {
		if dim == len(chunkShape) {
			// Calculate standard C-order flat indices
			chunkFlatIdx := 0
			for i, rc := range relCoords {
				chunkFlatIdx += rc * chunkStrides[i]
			}

			globalFlatIdx := 0
			for i, rc := range relCoords {
				globalFlatIdx += (chunkStartGlobal[i] + rc) * globalStrides[i]
			}

			chunkByteOffset := chunkFlatIdx * itemSize
			globalByteOffset := globalFlatIdx * itemSize

			// Bounds checking before copy to prevent panics on malformed chunks
			if chunkByteOffset+itemSize <= len(chunkData) && globalByteOffset+itemSize <= len(globalBuffer) {
				copy(globalBuffer[globalByteOffset:globalByteOffset+itemSize], chunkData[chunkByteOffset:chunkByteOffset+itemSize])
			}
			return
		}

		for i := 0; i < chunkShape[dim]; i++ {
			relCoords[dim] = i
			copyElements(dim+1, relCoords)
		}
	}

	relCoords := make([]int, len(chunkShape))
	copyElements(0, relCoords)

	return nil
}

// ReadRegion reads an N-dimensional region of the Zarr array.
func (r *Reader) ReadRegion(ctx context.Context, start, shape []int) ([]byte, error) {
	if len(start) != len(r.meta.Shape) || len(shape) != len(r.meta.Shape) {
		return nil, fmt.Errorf("start and shape must match array dimensionality")
	}

	// Validate bounds
	for i := range r.meta.Shape {
		if start[i] < 0 || shape[i] <= 0 || start[i]+shape[i] > r.meta.Shape[i] {
			return nil, fmt.Errorf("region out of bounds at dimension %d", i)
		}
	}

	// Calculate item size
	_, itemSize, err := ParseDType(r.meta.DType)
	if err != nil {
		return nil, fmt.Errorf("invalid dtype: %w", err)
	}

	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}
	out := make([]byte, totalElements*itemSize)

	if len(r.meta.Shape) == 0 {
		return r.ReadChunk(ctx, []int{})
	}

	minChunk := make([]int, len(start))
	maxChunk := make([]int, len(start))
	for i := range start {
		minChunk[i] = start[i] / r.meta.Chunks[i]
		maxChunk[i] = (start[i] + shape[i] - 1) / r.meta.Chunks[i]
	}

	dstStrides := strides(shape)
	chunkStrides := strides(r.meta.Chunks)

	var iterateChunks func(dim int, currentChunkCoords []int) error
	iterateChunks = func(dim int, currentChunkCoords []int) error {
		if dim == len(minChunk) {
			chunkData, err := r.ReadChunk(ctx, currentChunkCoords)
			if err != nil {
				return err
			}

			copyShape := make([]int, len(r.meta.Shape))
			srcOffset := make([]int, len(r.meta.Shape))
			dstOffset := make([]int, len(r.meta.Shape))

			for i := range r.meta.Shape {
				chunkStartGlobal := currentChunkCoords[i] * r.meta.Chunks[i]
				chunkEndGlobal := chunkStartGlobal + r.meta.Chunks[i]
				if chunkEndGlobal > r.meta.Shape[i] {
					chunkEndGlobal = r.meta.Shape[i]
				}

				reqStartGlobal := start[i]
				reqEndGlobal := start[i] + shape[i]

				intersectStart := max(chunkStartGlobal, reqStartGlobal)
				intersectEnd := min(chunkEndGlobal, reqEndGlobal)

				if intersectStart >= intersectEnd {
					return nil
				}

				copyShape[i] = intersectEnd - intersectStart
				srcOffset[i] = intersectStart - chunkStartGlobal
				dstOffset[i] = intersectStart - reqStartGlobal
			}

			copyND(out, dstStrides, dstOffset, chunkData, chunkStrides, srcOffset, copyShape, itemSize)
			return nil
		}

		for i := minChunk[dim]; i <= maxChunk[dim]; i++ {
			currentChunkCoords[dim] = i
			if err := iterateChunks(dim+1, currentChunkCoords); err != nil {
				return err
			}
		}
		return nil
	}

	coords := make([]int, len(minChunk))
	if err := iterateChunks(0, coords); err != nil {
		return nil, err
	}

	return out, nil
}

// copyND recursively copies n-dimensional data from src to dst.
func copyND(
	dst []byte, dstStrides, dstOffset []int,
	src []byte, srcStrides, srcOffset []int,
	copyShape []int, itemSize int,
) {
	if len(copyShape) == 0 {
		// 0D scalar array: exactly one element
		copy(dst[:itemSize], src[:itemSize])
		return
	}

	startSrcIdx := 0
	startDstIdx := 0
	for i := range copyShape {
		startSrcIdx += srcOffset[i] * srcStrides[i]
		startDstIdx += dstOffset[i] * dstStrides[i]
	}

	var iterate func(dim int, currentSrcIdx, currentDstIdx int)
	iterate = func(dim int, currentSrcIdx, currentDstIdx int) {
		// Optimization: bulk copy for the innermost contiguous dimension
		if dim == len(copyShape)-1 {
			n := copyShape[dim]
			if srcStrides[dim] == 1 && dstStrides[dim] == 1 {
				byteLen := n * itemSize
				srcStart := currentSrcIdx * itemSize
				dstStart := currentDstIdx * itemSize
				copy(dst[dstStart:dstStart+byteLen], src[srcStart:srcStart+byteLen])
				return
			}
			// Fallback for non-contiguous last dimension
			for i := 0; i < n; i++ {
				srcStart := (currentSrcIdx + i*srcStrides[dim]) * itemSize
				dstStart := (currentDstIdx + i*dstStrides[dim]) * itemSize
				copy(dst[dstStart:dstStart+itemSize], src[srcStart:srcStart+itemSize])
			}
			return
		}

		for i := 0; i < copyShape[dim]; i++ {
			iterate(dim+1, currentSrcIdx+i*srcStrides[dim], currentDstIdx+i*dstStrides[dim])
		}
	}
	iterate(0, startSrcIdx, startDstIdx)
}


func (r *Reader) Metadata() *Metadata {
	return r.meta
}

// Close closes the reader.
func (r *Reader) Close() error {
	return r.bucket.Close()
}
