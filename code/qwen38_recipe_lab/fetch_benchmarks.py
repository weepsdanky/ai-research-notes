"""Fetch bounded additional evaluation datasets at catalog-pinned revisions.

This is evaluation data, never an expansion of the training corpus. The default
batch is capped at 48MB; existing sources plus this batch stay below 64MB.
"""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path('data/eval-source')
FAMILIES = {
    'edinburgh-dawg/mmlu-redux': ('redux', ('.arrow',), True),
    'fxmarty/mmlu-redux-2.0-ok': ('redux2-ok', ('.arrow',), True),
    'm-a-p/SuperGPQA': ('supergpqa', ('.jsonl',), True),
    'EleutherAI/hendrycks_math': ('math', ('.parquet',), True),
    'CohereForAI/include-base-44': ('include', ('.parquet',), True),
    'evalplus/humanevalplus': ('humanevalplus', ('.parquet',), False),
    'evalplus/mbppplus': ('mbppplus', ('.parquet',), False),
}


def read(url, cap):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as response:
                raw = response.read(cap+1)
            if len(raw) > cap:
                raise ValueError('Evaluation byte cap exceeded')
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def main():
    lock = Path('eval_sources.lock.json')
    previous = json.loads(lock.read_text())
    by_name = {s['name']: s for s in previous}
    plan = []
    for entry in json.loads(Path('evaluation_catalog.json').read_text()):
        repo = entry['dataset']
        if repo not in FAMILIES:
            continue
        prefix, suffixes, card = FAMILIES[repo]
        tree = read(f"https://huggingface.co/api/datasets/{repo}/tree/{entry['revision']}?recursive=true", 2_000_000)
        for item in json.loads(tree):
            path = item['path']
            if item['type'] != 'file' or not (path.endswith(suffixes) or card and path in ('README.md', 'LICENSE')):
                continue
            if item['size'] > 32_000_000:
                raise ValueError('Individual evaluation file exceeds 32MB')
            plan.append(dict(name=prefix+'-'+path.replace('/', '-'), repo=repo,
                             revision=entry['revision'], bytes=item['size'],
                             upstream_sha256=item.get('lfs', {}).get('oid'),
                             url=f"https://huggingface.co/datasets/{repo}/resolve/{entry['revision']}/{quote(path)}"))
    if sum(s['bytes'] for s in plan) > 48_000_000:
        raise ValueError('Batch cap exceeded before download')
    union = dict(by_name)
    union.update({s['name']: s for s in plan})
    if sum(s['bytes'] for s in union.values()) > 64_000_000:
        raise ValueError('Combined evaluation source cap exceeded before download')
    ROOT.mkdir(parents=True, exist_ok=True)

    def fetch(spec):
        target = ROOT/spec['name']
        old = by_name.get(spec['name'])
        expected = old['sha256'] if old else spec['upstream_sha256']
        if expected and target.exists():
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == expected:
                    return old or dict(spec, sha256=expected)
        raw = read(spec['url'], spec['bytes'])
        if len(raw) != spec['bytes']:
            raise ValueError('Evaluation source length mismatch')
        digest = hashlib.sha256(raw).hexdigest()
        if expected and digest != expected:
            raise ValueError('Locked evaluation source drift')
        temp = target.with_suffix(target.suffix+'.tmp')
        temp.write_bytes(raw)
        temp.replace(target)
        print(spec['name'], len(raw), flush=True)
        return dict(spec, sha256=digest)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, plan))
    by_name.update({s['name']: s for s in fetched})
    temp = lock.with_suffix('.json.tmp')
    temp.write_text(json.dumps(list(by_name.values()), indent=2)+'\n')
    temp.replace(lock)
    print('Verified total bytes:', sum(s['bytes'] for s in by_name.values()), flush=True)


if __name__ == '__main__':
    main()
