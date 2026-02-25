import zarr
import numpy as np
import numcodecs
import os
import shutil

# Generate a basic 4x4 array with 2x2 chunks
# The exact values will be 0.0 to 15.0 to make testing predictable
data = np.arange(16, dtype=np.float32).reshape((4, 4))

def create_variation(name, compressor=None, filters=None):
    path = os.path.join("data", f"{name}.zarr")
    if os.path.exists(path):
        shutil.rmtree(path)
    
    z = zarr.open(
        path,
        mode='w',
        zarr_format=2,
        shape=(4, 4),
        chunks=(2, 2),
        dtype='<f4',
        compressor=compressor,
        filters=filters,
        order='C'
    )
    z[:] = data
    print(f"Generated {name}.zarr")


def main():
    # 1. Uncompressed
    create_variation("uncompressed")

    # 2. Zlib (Gzip)
    create_variation("zlib", compressor=numcodecs.Zlib(level=1))

    # 3. Blosc LZ4
    create_variation("blosc_lz4", compressor=numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE))

    # 4. Blosc ZSTD
    create_variation("blosc_zstd", compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.NOSHUFFLE))

    # 5. Blosc ZSTD with Shuffle
    create_variation("blosc_zstd_shuffle", compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE))

    # 6. Blosc ZSTD with BitShuffle
    create_variation("blosc_zstd_bitshuffle", compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.BITSHUFFLE))


if __name__ == "__main__":
    main()