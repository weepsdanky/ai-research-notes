"""Read-only Windows GGUF bandwidth probe; bypasses the Windows file cache."""
import ctypes
from ctypes import wintypes as w
import json
import random
import time
from pathlib import Path


def offsets(size, block, count, random_reads):
    assert block % 4096 == 0 and count > 0 and size >= block * count
    if random_reads:
        rng = random.Random(20261009)
        return [rng.randrange(size // block) * block for _ in range(count)]
    return [i * block for i in range(count)]


def main():
    root = Path(r'D:\ai-research\low-vram-inference')
    target = root / 'models/deepseek-q4/DeepSeek-V4-Flash-UD-Q4_K_XL-00004-of-00005.gguf'
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    k.CreateFileW.restype = w.HANDLE
    k.VirtualAlloc.argtypes = [w.LPVOID, ctypes.c_size_t, w.DWORD, w.DWORD]
    k.VirtualAlloc.restype = w.LPVOID
    k.SetFilePointerEx.argtypes = [w.HANDLE, ctypes.c_longlong, w.LPVOID, w.DWORD]
    k.ReadFile.argtypes = [w.HANDLE, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID]
    k.VirtualFree.argtypes = [w.LPVOID, ctypes.c_size_t, w.DWORD]
    k.CloseHandle.argtypes = [w.HANDLE]
    handle = k.CreateFileW(str(target), 0x80000000, 7, None, 3, 0x20000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = k.VirtualAlloc(None, 16 * 2**20, 0x3000, 4)
    if not buffer:
        k.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    try:
        for random_reads, block, count in [(False, 16 * 2**20, 256), (True, 4 * 2**20, 256)] * 3:
            latencies = []
            began = time.perf_counter()
            for offset in offsets(target.stat().st_size, block, count, random_reads):
                if not k.SetFilePointerEx(handle, offset, None, 0):
                    raise ctypes.WinError(ctypes.get_last_error())
                n = w.DWORD()
                t = time.perf_counter()
                if not k.ReadFile(handle, buffer, block, ctypes.byref(n), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                assert n.value == block
                latencies.append(time.perf_counter() - t)
            elapsed = time.perf_counter() - began
            row = {'random': random_reads, 'block_bytes': block, 'count': count,
                   'bytes': block * count, 'seconds': elapsed,
                   'GB_per_s': block * count / elapsed / 1e9, 'read_seconds': latencies}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'read_seconds'}), flush=True)
    finally:
        k.VirtualFree(buffer, 0, 0x8000)
        k.CloseHandle(handle)
    (root / 'results/disk-read-bandwidth.json').write_text(json.dumps({'target': str(target),
        'cache': 'FILE_FLAG_NO_BUFFERING; device cache uncontrolled', 'runs': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
