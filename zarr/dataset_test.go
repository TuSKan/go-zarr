package zarr_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/TuSKan/zarr-gomlx/zarr"
	"github.com/klauspost/compress/zstd"
	"github.com/stretchr/testify/require"
	_ "gocloud.dev/blob/fileblob"
)

func TestDataset_NextBatch(t *testing.T) {
	// 1. Setup temporary Zarr dataset
	tmpDir := t.TempDir()

	// Metadata: Shape=[10, 2], Chunks=[5, 2], DType="<f4" (float32)
	meta := zarr.Metadata{
		ZarrFormat: 2,
		Shape:      []int{10, 2},
		Chunks:     []int{5, 2},
		DType:      "<f4",
	}

	metaBytes, err := json.Marshal(meta)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, ".zarray"), metaBytes, 0644))

	// Create chunks: 0.0 and 1.0
	// Chunk 0.0 covers rows 0-4
	// Chunk 1.0 covers rows 5-9
	createFloat32Chunk(t, tmpDir, "0.0", []float32{
		0, 1,
		2, 3,
		4, 5,
		6, 7,
		8, 9,
	})
	createFloat32Chunk(t, tmpDir, "1.0", []float32{
		10, 11,
		12, 13,
		14, 15,
		16, 17,
		18, 19,
	})

	// 2. Open Dataset
	ctx := context.Background()
	ds, err := zarr.NewDataset(ctx, "file://"+tmpDir)
	require.NoError(t, err)

	// 3. Read Batch 1 (size 3) -> Rows 0, 1, 2
	batch1, err := ds.NextBatch(ctx, 3)
	require.NoError(t, err)
	require.Equal(t, []int{3, 2}, batch1.Shape().Dimensions)
	require.Equal(t, [][]float32{{0, 1}, {2, 3}, {4, 5}}, batch1.Value().([][]float32))

	// 4. Read Batch 2 (size 3) -> Rows 3, 4, 5 (Crosses chunk boundary)
	batch2, err := ds.NextBatch(ctx, 3)
	require.NoError(t, err)
	require.Equal(t, []int{3, 2}, batch2.Shape().Dimensions)
	require.Equal(t, [][]float32{{6, 7}, {8, 9}, {10, 11}}, batch2.Value().([][]float32))

	// 5. Read Batch 3 (size 4) -> Rows 6, 7, 8, 9 (Remaining)
	batch3, err := ds.NextBatch(ctx, 4)
	require.NoError(t, err)
	require.Equal(t, []int{4, 2}, batch3.Shape().Dimensions)
	require.Equal(t, [][]float32{{12, 13}, {14, 15}, {16, 17}, {18, 19}}, batch3.Value().([][]float32))

	// 6. EOF
	_, err = ds.NextBatch(ctx, 1)
	require.ErrorIs(t, err, io.EOF)
}

func createFloat32Chunk(t *testing.T, dir, name string, data []float32) {
	f, err := os.Create(filepath.Join(dir, name))
	require.NoError(t, err)
	defer f.Close()

	for _, v := range data {
		err := binary.Write(f, binary.LittleEndian, v)
		require.NoError(t, err)
	}
}

func TestDataset_NextBatch_Zstd(t *testing.T) {
	// 1. Setup temporary Zarr dataset
	tmpDir := t.TempDir()

	// Metadata: Shape=[10, 2], Chunks=[5, 2], DType="<f4" (float32), Compressor="zstd"
	meta := zarr.Metadata{
		ZarrFormat: 2,
		Shape:      []int{10, 2},
		Chunks:     []int{5, 2},
		DType:      "<f4",
		Compressor: &zarr.CompressorConfig{ID: "zstd"},
	}

	metaBytes, err := json.Marshal(meta)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, ".zarray"), metaBytes, 0644))

	// Create compressed chunks
	createCompressedFloat32Chunk(t, tmpDir, "0.0", []float32{
		0, 1,
		2, 3,
		4, 5,
		6, 7,
		8, 9,
	})
	createCompressedFloat32Chunk(t, tmpDir, "1.0", []float32{
		10, 11,
		12, 13,
		14, 15,
		16, 17,
		18, 19,
	})

	// 2. Open Dataset
	ctx := context.Background()
	ds, err := zarr.NewDataset(ctx, "file://"+tmpDir)
	require.NoError(t, err)

	// 3. Read Batch (size 10) -> All rows
	batch, err := ds.NextBatch(ctx, 10)
	require.NoError(t, err)
	require.Equal(t, []int{10, 2}, batch.Shape().Dimensions)

	expected := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		expected[i] = []float32{float32(i * 2), float32(i*2 + 1)}
	}
	require.Equal(t, expected, batch.Value().([][]float32))
}

func createCompressedFloat32Chunk(t *testing.T, dir, name string, data []float32) {
	// 1. Encode data to bytes
	var buf []byte
	for _, v := range data {
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, math.Float32bits(v))
		buf = append(buf, b...)
	}

	// 2. Compress
	var compressed []byte
	encoder, err := zstd.NewWriter(nil)
	require.NoError(t, err)
	compressed = encoder.EncodeAll(buf, nil)
	encoder.Close()

	// 3. Write to file
	require.NoError(t, os.WriteFile(filepath.Join(dir, name), compressed, 0644))
}
