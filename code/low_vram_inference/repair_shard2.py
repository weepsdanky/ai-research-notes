"""Repair the inherited prefix in place, then require the full publisher hash."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from download_deepseek import ROOT, check_range


def main():
    if os.name != 'nt':
        raise RuntimeError('Run on the Windows weight host')
    manifest = json.loads(Path(__file__).with_name('deepseek-q4-manifest.json').read_text())
    item = manifest['files'][1]
    target = ROOT / Path(item['path']).name
    if target.stat().st_size != item['size']:
        raise RuntimeError('Expected the diagnosed complete shard')
    # The inherited curl prefix precedes native bounded Range downloads.
    # This bound is a repair hypothesis, never a substitute for full SHA256.
    limit = min(8 * 1024**3, item['size'])
    evidence = ROOT.parent.parent / 'results/shard2-repair.jsonl'
    with target.open('r+b') as output, evidence.open('a') as log:
        for start in range(0, limit, 64 * 1024**2):
            end = min(start + 64 * 1024**2, limit) - 1
            for attempt in range(21):
                try:
                    url = 'https://huggingface.co/{}/resolve/{}/{}?repair={}&attempt={}'.format(
                        manifest['repo'], manifest['revision'], item['path'], start, attempt)
                    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206:
                            raise RuntimeError('Expected HTTP 206')
                        count = check_range(response.headers.get('Content-Range'), start, end, item['size'])
                        data = response.read(count)
                    if len(data) != end - start + 1:
                        raise OSError('Incomplete repair range')
                    break
                except OSError:
                    if attempt == 20:
                        raise
                    time.sleep(min(3 * (attempt + 1), 30))
            output.seek(start)
            old = output.read(len(data))
            changed = old != data
            if changed:
                output.seek(start)
                output.write(data)
                output.flush()
            record = {'offset': start, 'bytes': len(data), 'changed': changed,
                      'before_sha256': hashlib.sha256(old).hexdigest(),
                      'source_sha256': hashlib.sha256(data).hexdigest()}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    actual = digest.hexdigest()
    result = {'expected_sha256': item['sha256'], 'actual_sha256': actual,
              'passed': actual == item['sha256'], 'repaired_prefix_bytes': limit}
    evidence.with_suffix('.result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Prefix repair insufficient; full publisher hash still differs')


if __name__ == '__main__':
    main()
