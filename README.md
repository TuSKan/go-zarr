# go-zarr

A pure-Go reader for [Zarr](https://zarr.readthedocs.io/en/stable/) V2 datasets.

`go-zarr` provides an efficient, native Go implementation for reading N-dimensional typed arrays from Zarr stores. It supports reading both local arrays and remote cloud/HTTP datasets, with an emphasis on production-grade features like subsetting (`ReadRegion`), Blosc decompression, and `zlib`/`gzip` fallback.

## Features

- **Pure Go**: No CGO dependencies required. Includes a native Go implementation of [Blosc](https://github.com/Blosc/c-blosc) decompression via a specialized fork (`github.com/mrjoshuak/go-blosc`).
- **Flexible Storage**: Uses [`gocloud.dev/blob`](https://gocloud.dev/howto/blob/) to seamlessly support local directories (`file://`), S3 (`s3://`), GCS (`gs://`), Azure (`azblob://`), HTTP/HTTPS, and more.
- **N-Dimensional Subsetting**: Read the entire array (`ReadFull`) or specific N-dimensional contiguous regions (`ReadRegion`).
- **Diverse Types**: Safely parses `numpy`-style dtypes (e.g., `<f4`, `|i8`, `<u2`, `|b1`) and handles endianness appropriately.
- **Production Proven**: Tested against real-world OME-NGFF and ERA5 Zarr V2 datasets containing thousands of chunks.

## Installation

```bash
go get github.com/TuSKan/go-zarr
```

## Usage

### Reading the Entire Dataset

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/TuSKan/go-zarr"
	// Import the gocloud blob drivers you need
	_ "gocloud.dev/blob/fileblob" 
)

func main() {
	ctx := context.Background()

	// path can be any gocloud URI: 
	// file:///path/to/data.zarr, s3://my-bucket/data.zarr, gs://my-bucket/...
	reader, err := zarr.NewReader(ctx, "file:///path/to/my_array.zarr")
	if err != nil {
		log.Fatalf("failed to open zarr reader: %v", err)
	}
	defer reader.Close()

	// Read the metadata
	meta := reader.Metadata()
	fmt.Printf("Shape: %v, Chunks: %v, DType: %s\n", meta.Shape, meta.Chunks, meta.DType)

	// Read the entire array
	data, err := reader.ReadFull(ctx)
	if err != nil {
		log.Fatalf("failed to read full array: %v", err)
	}

	fmt.Printf("Read %d bytes successfully.\n", len(data))
}
```

### Reading an N-Dimensional Region

If you only need a portion of a large array, you can use `ReadRegion` to fetch and decompress only the intersecting chunks.

```go
// ... 
// Inside the main block, after instantiating NewReader:

// Read a 3D sub-region starting at coordinates [0, 100, 100] 
// with a target shape of [10, 50, 50]
start := []int{0, 100, 100}
shape := []int{10, 50, 50}

regionData, err := reader.ReadRegion(ctx, start, shape)
if err != nil {
	log.Fatalf("failed to read region: %v", err)
}

fmt.Printf("Read %d bytes for the specified region.\n", len(regionData))
```

## Testing

The testing suite contains:
1. Static Zarr Variations: Dozens of locally generated Zarr arrays with different compressors (Blosc variations, Zlib) and data types.
2. Real-World Datasets: Tests that download partial Zarrs over HTTP from production repositories (e.g., OME-NGFF microscope images) to verify real-world behavior.

To run tests:
```bash
go test ./...
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
