"""Restore bounded locked MultiPL-E inputs, including publisher-derived Python."""
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


def checked(path, spec):
    return path.exists() and path.stat().st_size == spec['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']


def fetch(path, spec):
    if checked(path, spec):return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec['url'], timeout=60) as response:
        data = response.read(spec['bytes']+1)
    if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
        raise ValueError('Locked download mismatch: '+str(path))
    temporary = path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)


def main():
    lab = Path(__file__).resolve().parent
    os.chdir(lab)
    specs = json.loads(Path('multipl_sources.lock.json').read_text())
    if sum(s['bytes'] for s in specs) > 16_000_000:raise ValueError('Data cap')
    archive_spec = json.loads(Path('multipl_archive.lock.json').read_text())
    archive = Path('.cache/multipl-source.tar.gz');fetch(archive, archive_spec)
    root = lab/'.cache/multipl-source'/('MultiPL-E-'+archive_spec['revision'])
    # Re-extract only a SHA-verified archive, with a bounded, traversal-safe filter.
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if sum(m.size for m in members) > archive_spec['expanded_cap']:raise ValueError('Expanded archive cap')
        if any(m.issym() or m.islnk() or not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe archive member')
        stream.extractall(root.parent, members=members, filter='data')
    for spec in specs:
        path = lab/'data/multipl'/spec['name']
        if spec['language'] != 'py':fetch(path, spec);continue
        os.chdir(root/'dataset_builder');sys.path.insert(0, str(Path.cwd()))
        from generic_translator import list_originals, translate_prompt_and_tests, get_stop_from_translator
        from humaneval_to_py import Translator
        translator = Translator();rows = []
        originals = list_originals('../datasets/originals-with-cleaned-doctests')
        if len(originals) != 161:raise ValueError('Unexpected cleaned-originals coverage')
        for original in originals.values():
            prompt, tests = translate_prompt_and_tests(original, translator, 'transform', 'reworded', add_canonical_to_prompt=False, panic_on_test_fail=True)
            rows.append(dict(name=original.stem, language='py', prompt=prompt, tests=tests,
                             stop_tokens=get_stop_from_translator(translator), doctests='transform', prompt_terminology='reworded'))
        rows.sort(key=lambda r:int(r['name'].split('_')[1]))
        data = ''.join(json.dumps(r)+'\n' for r in rows).encode()
        if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:raise ValueError('Python derivation drift')
        path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data)
        os.chdir(lab)
    print('Verified eight complete configurations, 1277 tasks')


if __name__ == '__main__':main()
