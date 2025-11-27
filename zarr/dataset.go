package zarr

import (
	"context"
	"encoding/binary"
	"encoding/json/v2"
	"fmt"
	"io"
	"math"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/klauspost/compress/zstd"
	"gocloud.dev/blob"
	"gocloud.dev/gcerrors"
)

// Dataset handles reading Zarr arrays in batches.
type Dataset struct {
	bucket       *blob.Bucket
	meta         *Metadata
	CurrentIndex int
}

// NewDataset creates a new Dataset for the given base path.
func NewDataset(ctx context.Context, path string) (*Dataset, error) {
	bucket, err := blob.OpenBucket(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to open bucket: %w", err)
	}

	ds := &Dataset{bucket: bucket, meta: &Metadata{}}
	if err := ds.readMetadata(ctx); err != nil {
		bucket.Close()
		return nil, err
	}
	return ds, nil
}

// readMetadata reads the metadata for the Zarr array.
func (d *Dataset) readMetadata(ctx context.Context) error {
	// LoadMetadata reads and parses the .zarray metadata from the given path.
	reader, err := d.bucket.NewReader(ctx, ".zarray", nil)
	if err != nil {
		return fmt.Errorf("failed to open .zarray: %w", err)
	}
	defer reader.Close()

	if err := json.UnmarshalRead(reader, d.meta); err != nil {
		return fmt.Errorf("failed to decode metadata: %w", err)
	}

	if d.meta.ZarrFormat != 2 {
		return fmt.Errorf("unsupported zarr_format: %d, expected 2", d.meta.ZarrFormat)
	}

	return nil
}

// NextBatch reads the next batch of size batchSize.
// Returns io.EOF if there is no more data.
func (d *Dataset) NextBatch(ctx context.Context, batchSize int) (*tensors.Tensor, error) {
	if d.CurrentIndex >= d.meta.Shape[0] {
		return nil, io.EOF
	}

	start := d.CurrentIndex
	end := start + batchSize
	if end > d.meta.Shape[0] {
		end = d.meta.Shape[0]
	}
	actualBatchSize := end - start

	// Allocate buffer for the batch
	// Batch shape: [actualBatchSize, Shape[1], Shape[2]...]
	batchShape := make([]int, len(d.meta.Shape))
	batchShape[0] = actualBatchSize
	copy(batchShape[1:], d.meta.Shape[1:])

	totalElements := 1
	for _, dim := range batchShape {
		totalElements *= dim
	}

	var data any
	switch d.meta.DType {
	case "<f4":
		data = make([]float32, totalElements)
	case "<i4":
		data = make([]int32, totalElements)
	case "<i8":
		data = make([]int64, totalElements)
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", d.meta.DType)
	}

	// Identify chunks that overlap with [start, end) in dim 0
	chunkSize0 := d.meta.Chunks[0]
	startChunk0 := start / chunkSize0
	endChunk0 := (end - 1) / chunkSize0

	// We need to iterate over all chunks in other dimensions as well
	chunkGridShape := make([]int, len(d.meta.Shape))
	for i := range d.meta.Shape {
		chunkGridShape[i] = int(math.Ceil(float64(d.meta.Shape[i]) / float64(d.meta.Chunks[i])))
	}

	subGridStart := make([]int, len(chunkGridShape))
	subGridEnd := make([]int, len(chunkGridShape))
	copy(subGridEnd, chunkGridShape) // Default to full range

	subGridStart[0] = startChunk0
	subGridEnd[0] = endChunk0 + 1 // Exclusive

	// Iterate chunks
	err := iterateSubGrid(subGridStart, subGridEnd, func(chunkIndices []int) error {
		key := ChunkKey(chunkIndices, ".")

		// Read chunk file
		reader, err := d.bucket.NewReader(ctx, key, nil)
		if err != nil {
			if gcerrors.Code(err) == gcerrors.NotFound {
				// Chunk not found, treat as empty (zeros)
				// TODO: Handle fill value
				return nil
			}
			return fmt.Errorf("failed to open chunk %s: %w", key, err)
		}
		defer reader.Close()

		chunkBytes, err := io.ReadAll(reader)
		if err != nil {
			return fmt.Errorf("failed to read chunk %s: %w", key, err)
		}

		// Decompress if needed
		if d.meta.Compressor != nil {
			switch d.meta.Compressor.ID {
			case "zstd":
				decoder, err := zstd.NewReader(nil)
				if err != nil {
					return fmt.Errorf("failed to create zstd reader: %w", err)
				}
				defer decoder.Close()
				chunkBytes, err = decoder.DecodeAll(chunkBytes, nil)
				if err != nil {
					return fmt.Errorf("failed to decompress chunk %s: %w", key, err)
				}
			case "blosc":
				return fmt.Errorf("blosc compression not yet supported")
			default:
				return fmt.Errorf("unsupported compressor: %s", d.meta.Compressor.ID)
			}
		}

		// Copy relevant part of chunk to batch buffer
		return copyChunkToBatch(data, chunkBytes, chunkIndices, d.meta, start, end, batchShape)
	})
	if err != nil {
		return nil, err
	}

	d.CurrentIndex = end
	switch v := data.(type) {
	case []float32:
		return tensors.FromFlatDataAndDimensions(v, batchShape...), nil
	case []int32:
		return tensors.FromFlatDataAndDimensions(v, batchShape...), nil
	case []int64:
		return tensors.FromFlatDataAndDimensions(v, batchShape...), nil
	default:
		return nil, fmt.Errorf("unexpected data type: %T", data)
	}
}

// iterateSubGrid iterates from start (inclusive) to end (exclusive) in each dimension.
func iterateSubGrid(start, end []int, fn func(indices []int) error) error {
	if len(start) == 0 {
		return fn([]int{})
	}
	indices := make([]int, len(start))
	copy(indices, start)

	for {
		if err := fn(indices); err != nil {
			return err
		}

		// Increment
		i := len(start) - 1
		for ; i >= 0; i-- {
			indices[i]++
			if indices[i] < end[i] {
				break
			}
			indices[i] = start[i] // Reset to start, not 0
		}
		if i < 0 {
			break
		}
	}
	return nil
}

func copyChunkToBatch(batchData any, chunkBytes []byte, chunkIndices []int, meta *Metadata, batchStartGlobal, batchEndGlobal int, batchShape []int) error {
	// Chunk global range
	chunkStart := make([]int, len(meta.Shape))
	chunkEnd := make([]int, len(meta.Shape))
	chunkShape := make([]int, len(meta.Shape))

	for i, idx := range chunkIndices {
		chunkStart[i] = idx * meta.Chunks[i]
		chunkEnd[i] = chunkStart[i] + meta.Chunks[i]
		if chunkEnd[i] > meta.Shape[i] {
			chunkEnd[i] = meta.Shape[i]
		}
		chunkShape[i] = chunkEnd[i] - chunkStart[i]
	}

	// Intersection with batch range
	// Batch covers global range [batchStartGlobal, batchEndGlobal) in dim 0
	// and [0, Shape[i]) in other dims.

	intersectStart := make([]int, len(meta.Shape))
	intersectEnd := make([]int, len(meta.Shape))

	// Dim 0
	intersectStart[0] = max(batchStartGlobal, chunkStart[0])
	intersectEnd[0] = min(batchEndGlobal, chunkEnd[0])

	// Other dims
	for i := 1; i < len(meta.Shape); i++ {
		intersectStart[i] = chunkStart[i]
		intersectEnd[i] = chunkEnd[i]
	}

	if intersectStart[0] >= intersectEnd[0] {
		return nil // No overlap
	}

	// Shape of the intersection volume
	intersectShape := make([]int, len(meta.Shape))
	for i := range meta.Shape {
		intersectShape[i] = intersectEnd[i] - intersectStart[i]
	}

	// Strides
	// Batch strides
	batchStrides := make([]int, len(batchShape))
	stride := 1
	for i := len(batchShape) - 1; i >= 0; i-- {
		batchStrides[i] = stride
		stride *= batchShape[i]
	}

	// Chunk strides
	chunkStrides := make([]int, len(chunkShape))
	stride = 1
	for i := len(chunkShape) - 1; i >= 0; i-- {
		chunkStrides[i] = stride
		stride *= chunkShape[i]
	}

	// Cast buffers
	var f32Batch []float32
	var i32Batch []int32
	var i64Batch []int64
	elementSize := 4

	switch v := batchData.(type) {
	case []float32:
		f32Batch = v
	case []int32:
		i32Batch = v
	case []int64:
		i64Batch = v
		elementSize = 8
	}
	if meta.DType == "<f8" || meta.DType == "<i8" {
		elementSize = 8
	}

	// Iterate over the intersection volume
	return iterateSubGrid(make([]int, len(meta.Shape)), intersectShape, func(relIndices []int) error {
		// Calculate global coords of this element
		globalCoords := make([]int, len(meta.Shape))
		for i := range relIndices {
			globalCoords[i] = intersectStart[i] + relIndices[i]
		}

		// 1. Source Index (in Chunk)
		chunkOffset := 0
		for i := range globalCoords {
			chunkOffset += (globalCoords[i] - chunkStart[i]) * chunkStrides[i]
		}

		byteOffset := chunkOffset * elementSize
		if byteOffset+elementSize > len(chunkBytes) {
			return fmt.Errorf("chunk index out of bounds")
		}

		// 2. Dest Index (in Batch)
		batchIndex := (globalCoords[0] - batchStartGlobal) * batchStrides[0]
		for i := 1; i < len(globalCoords); i++ {
			batchIndex += globalCoords[i] * batchStrides[i]
		}

		// Copy value
		switch meta.DType {
		case "<f4":
			bits := binary.LittleEndian.Uint32(chunkBytes[byteOffset:])
			f32Batch[batchIndex] = math.Float32frombits(bits)
		case "<i4":
			val := int32(binary.LittleEndian.Uint32(chunkBytes[byteOffset:]))
			i32Batch[batchIndex] = val
		case "<i8":
			val := int64(binary.LittleEndian.Uint64(chunkBytes[byteOffset:]))
			i64Batch[batchIndex] = val
		}
		return nil
	})
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
