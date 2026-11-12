"""Fetch selected complete MMMLU languages, within a separate 128MB cache."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen


def receipt(language, digest):
    with Path('data/mmmlu/.source.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        file = Path('mmmlu_sources.lock.json');rows = json.loads(file.read_text())
        row = next(s for s in rows if s['language'] == language)
        if row['sha256'] not in (None, digest):raise ValueError('MMMLU receipt drift')
        row['sha256'] = digest
        temp = file.with_suffix('.json.tmp');temp.write_text(json.dumps(rows, indent=2)+'\n');temp.replace(file)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--languages', nargs='+', required=True)
    a = p.parse_args()
    lock = json.loads(Path('mmmlu_sources.lock.json').read_text())
    available = {s['language']: s for s in lock}
    if any(lang not in available for lang in a.languages):p.error('Select published language codes: '+', '.join(available))
    plan = [available[lang] for lang in dict.fromkeys(a.languages)]
    if sum(s['bytes'] for s in plan) > 32_000_000 or sum(s['bytes'] for s in lock) > 128_000_000:
        raise ValueError('MMMLU invocation/cache budget exceeded')
    root = Path('data/mmmlu');root.mkdir(parents=True, exist_ok=True)
    for spec in plan:
        if spec['bytes'] > 20_000_000:raise ValueError('File budget exceeded')
        file = root/spec['name']
        if file.exists():
            raw = file.read_bytes()
            sha256 = hashlib.sha256(raw).hexdigest()
            valid = sha256 == spec['sha256'] if spec['sha256'] else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == spec['git_blob_sha1']
            if len(raw) == spec['bytes'] and valid:
                receipt(spec['language'], sha256)
                print('already verified', spec['language'], flush=True);continue
            raise ValueError('Existing MMMLU cache drift; preserve file for inspection')
        for attempt in range(3):
            try:
                with urlopen(spec['url'], timeout=60) as response:raw = response.read(spec['bytes']+1)
                break
            except OSError:
                if attempt == 2:raise
                time.sleep(attempt+1)
        sha256 = hashlib.sha256(raw).hexdigest()
        publisher_matches = sha256 == spec['sha256'] if spec['sha256'] else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == spec['git_blob_sha1']
        if len(raw) != spec['bytes'] or not publisher_matches:
            raise ValueError('MMMLU source mismatch')
        temp = file.with_suffix('.csv.tmp');temp.write_bytes(raw);temp.replace(file)
        receipt(spec['language'], sha256)
        print('verified', spec['language'], len(raw), flush=True)


if __name__ == '__main__':
    main()
