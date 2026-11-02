"""One pinned weight copy; Windows stdlib Range resume with two file workers."""
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RESERVE = 40_000_000_000
CHUNK = 512 * 1024 * 1024
ROOT = Path(r'D:\ai-research\low-vram-inference\models\deepseek-q4')


def remaining(files, root):
    return sum(max(0, f['size'] - (root / Path(f['path']).name).stat().st_size)
               if (root / Path(f['path']).name).exists() else f['size'] for f in files)


def check_range(header, start, end, total):
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', header or '')
    if not match:
        raise ValueError('Missing/invalid Content-Range')
    a, b, size = map(int, match.groups())
    if a != start or not a <= b <= end or size != total:
        raise ValueError('Wrong response range: ' + str(header))
    return b - a + 1


def download(item, manifest, stop):
    target = ROOT / Path(item['path']).name
    retries = 0
    while not target.exists() or target.stat().st_size < item['size']:
        if stop.is_set():
            raise RuntimeError('Another file failed; stopped')
        start = target.stat().st_size if target.exists() else 0
        end = min(start + CHUNK, item['size']) - 1
        url = 'https://huggingface.co/{}/resolve/{}/{}?download=true&offset={}&attempt={}'.format(
            manifest['repo'], manifest['revision'], item['path'], start, retries)
        print('Range', target.name, start, end, flush=True)
        request = urllib.request.Request(url, headers={'Range': 'bytes={}-{}'.format(start, end)})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise ValueError('Expected HTTP 206 before append')
                count = check_range(response.headers.get('Content-Range'), start, end, item['size'])
                with target.open('ab') as output:
                    while count:
                        if stop.is_set():
                            raise RuntimeError('Another file failed; stopped')
                        if shutil.disk_usage(ROOT).free < remaining(manifest['files'], ROOT) + RESERVE:
                            raise RuntimeError('Complete download would violate 40 GB reserve')
                        block = response.read(min(1024 * 1024, count))
                        if not block:
                            raise ConnectionError('Response ended early; saved prefix retained')
                        output.write(block)
                        count -= len(block)
            retries = 0
        except (OSError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (403, 408, 429, 500, 502, 503, 504):
                raise
            retries += 1
            if retries > 20:
                raise
            print('RETRY', target.name, type(error).__name__, 'attempt', retries, flush=True)
            stop.wait(min(3 * retries, 30))
    if target.stat().st_size != item['size']:
        raise RuntimeError('File size mismatch: ' + target.name)
    print('Hashing', target.name, flush=True)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            if stop.is_set():
                raise RuntimeError('Another file failed; stopped hashing')
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise RuntimeError('SHA256 mismatch; preserve file for diagnosis: ' + target.name)
    print('VERIFIED', target.name, digest.hexdigest(), flush=True)
    return dict(item, verified_sha256=digest.hexdigest(), mtime_ns=target.stat().st_mtime_ns)


def require_verified(root, manifest):
    proof = json.loads((root / 'verified.json').read_text())
    if proof.get('repo') != manifest['repo'] or proof.get('revision') != manifest['revision']:
        raise RuntimeError('Verification artifact has wrong checkpoint identity')
    records = {x['path']: x for x in proof.get('files', [])}
    if len(records) != len(manifest['files']) or len(proof['files']) != len(records):
        raise RuntimeError('Verification artifact has missing/duplicate shards')
    for item in manifest['files']:
        record = records.get(item['path'], {})
        target = root / Path(item['path']).name
        if (record.get('verified_sha256') != item['sha256'] or record.get('size') != item['size']
                or not target.is_file() or target.stat().st_size != item['size']
                or target.stat().st_mtime_ns != record.get('mtime_ns')):
            raise RuntimeError('Shard not verified or changed: ' + target.name)
    return proof


def main():
    if os.name != 'nt':
        raise RuntimeError('Run with Windows native Python; uses existing system HTTPS proxy')
    manifest = json.loads(Path(__file__).with_name('deepseek-q4-manifest.json').read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        target = ROOT / Path(item['path']).name
        if target.exists() and target.stat().st_size > item['size']:
            raise RuntimeError('Existing file exceeds expected size: ' + target.name)
    if shutil.disk_usage(ROOT).free < remaining(manifest['files'], ROOT) + RESERVE:
        raise RuntimeError('Insufficient space for complete model plus 40 GB reserve')
    stop = threading.Event()
    verified = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, item, manifest, stop) for item in manifest['files']]
        try:
            for future in concurrent.futures.as_completed(futures):
                verified.append(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    (ROOT / 'verified.json').write_text(json.dumps(dict(manifest, files=verified), indent=2))
    print('ALL FIVE SHARDS VERIFIED', flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [{'path': 'quant/a.gguf', 'size': 10}, {'path': 'quant/b.gguf', 'size': 20}]
            assert remaining(files, root) == 30
            (root / 'a.gguf').write_bytes(b'1234')
            assert remaining(files, root) == 26
            (root / 'b.gguf').write_bytes(b'x' * 20)
            assert remaining(files, root) == 6
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / 'a.gguf'
            shard.write_bytes(b'abc')
            sha = hashlib.sha256(b'abc').hexdigest()
            item = {'path': 'a.gguf', 'size': 3, 'sha256': sha}
            manifest = {'repo': 'test', 'revision': 'pinned', 'files': [item]}
            proof = dict(manifest, files=[dict(item, verified_sha256=sha, mtime_ns=shard.stat().st_mtime_ns)])
            (root / 'verified.json').write_text(json.dumps(proof))
            require_verified(root, manifest)
            shard.write_bytes(b'changed')
            try:
                require_verified(root, manifest)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Modified shard accepted')
        assert check_range('bytes 4-9/10', 4, 9, 10) == 6
        for bad in ('bytes 0-9/10', 'bytes 4-11/10', 'bytes 4-9/11', None):
            try:
                check_range(bad, 4, 9, 10)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe append range accepted')
        print('Space/resume and response-range self-check passed')
    elif '--check-verified' in sys.argv:
        manifest = json.loads(Path(__file__).with_name('deepseek-q4-manifest.json').read_text())
        require_verified(ROOT, manifest)
        print('Complete pinned verification artifact and unchanged shards confirmed')
    else:
        main()
