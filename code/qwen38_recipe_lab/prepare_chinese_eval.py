"""Fetch only the small, hash-locked public Chinese evaluation artifacts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

from qwen35_quality_data import digest, load


def fetch(item):
    path = Path(item['path'])
    root = Path('data/chinese-eval').resolve()
    assert path.resolve().is_relative_to(root)
    if path.exists():
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(item['url'], timeout=30) as stream:
        payload = stream.read(item['bytes']+1)
    assert len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256']
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory)/'verified'
        temporary.write_bytes(payload)
        os.link(temporary,path)


if __name__ == '__main__':
    files = load('chinese_eval_sources.lock.json')['files']
    total = sum(item['bytes'] for item in files)
    assert total < 8_000_000
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,files))
    print(f'Verified {len(files)} locked public files ({total:,} bytes).')
