"""RAM read/compare bandwidth calibration; 1 GiB allocation, no model access."""
import ctypes
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    libc = ctypes.CDLL('msvcrt' if os.name == 'nt' else None)
    libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcmp.restype = ctypes.c_int
    size = 128 * 2**20
    buffers = [ctypes.create_string_buffer(size) for _ in range(8)]
    for buffer in buffers:
        ctypes.memset(ctypes.addressof(buffer), 0x5a, size)
    def read(pair):
        a, b = buffers[2 * pair:2 * pair + 2]
        for _ in range(8):
            if libc.memcmp(a, b, size) != 0:
                raise RuntimeError('Full-buffer comparison failed')
    # A mismatch self-check verifies the function actually observes buffer data.
    buffers[1][size - 1] = b'\x00'
    assert libc.memcmp(buffers[0], buffers[1], size) != 0
    buffers[1][size - 1] = b'\x5a'
    rows = []
    for workers in (1, 2, 4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for repeat in range(3):
                began = time.perf_counter()
                list(pool.map(read, range(4)))
                elapsed = time.perf_counter() - began
                row = {'workers': workers, 'repeat': repeat, 'bytes_read': size * 2 * 8 * 4,
                       'seconds': elapsed, 'GB_per_s': size * 2 * 8 * 4 / elapsed / 1e9}
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(r'D:\ai-research\low-vram-inference\results\ram-read-bandwidth.json').write_text(
        json.dumps({'method': 'memcmp equal buffers; both inputs counted; CPU compare overhead included',
                    'allocated_bytes': size * 8, 'runs': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
