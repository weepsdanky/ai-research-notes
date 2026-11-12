"""Verify all generation helper hashes against one immutable local Git archive."""
import hashlib
import json
from pathlib import Path
import subprocess


def verify_generation_sources(identity, archive):
    prefix = subprocess.run(['git', 'rev-parse', '--show-prefix'], capture_output=True, text=True, check=True).stdout.strip()
    sources = {}
    for group in ('source_hashes', 'runtime_model_sources', 'chunk_sources'):
        sources.update(identity.get(group, {}) or {})
    if identity.get('prefix_cache'):
        sources['prefix_inference.py'] = identity['prefix_cache']['source_sha256']
    locks = json.loads(subprocess.run(['git', 'show', archive+':'+prefix+'sources.lock.json'], capture_output=True, check=True).stdout)
    for file, expected in sources.items():
        relative = Path(file)
        if relative.is_absolute():
            relative = relative.relative_to(Path.cwd())
        if relative.parts[:2] == ('.cache', 'references'):
            locked = next(row['sha256'] for row in locks if row['name'] == relative.name)
            with relative.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert expected == locked == actual
        else:
            raw = subprocess.run(['git', 'show', archive+':'+prefix+relative.as_posix()], capture_output=True, check=True).stdout
            assert hashlib.sha256(raw).hexdigest() == expected
    return sources
