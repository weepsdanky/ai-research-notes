"""Fetch small hash-locked source/tokenizer files; never model weights."""
import argparse,hashlib,json
from pathlib import Path
from urllib.request import urlopen,Request


def matches(spec,data):
    if len(data)!=spec['bytes'] or hashlib.sha256(data).hexdigest()!=spec['sha256']:
        return False
    blob=spec.get('git_blob_sha1')
    return not blob or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==blob


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--evaluation',action='store_true',help='Fetch all byte-locked evaluation source files instead')
    mode.add_argument('--long-context',action='store_true',help='Fetch small locked RULER/MRCR/chat sources, never dataset shards')
    mode.add_argument('--qwen-release',action='store_true',help='Fetch the pinned Qwen release README and report')
    args=parser.parse_args()
    root=Path(__file__).parent
    target=root/('.cache/long-context-sources' if args.long_context else 'data/eval-source' if args.evaluation else '.cache/references')
    target.mkdir(parents=True,exist_ok=True)
    lock='long_context_sources.lock.json' if args.long_context else 'eval_sources.lock.json' if args.evaluation else 'sources.lock.json'
    if args.qwen_release:
        lock='qwen_release_sources.lock.json'
    for spec in json.loads((root/lock).read_text()):
        file=root/spec['file'] if args.long_context else target/spec['name']
        if args.long_context and file.resolve().parent!=target.resolve():raise ValueError('Unexpected long-context cache path')
        if file.exists() and matches(spec,file.read_bytes()):
            continue
        with urlopen(Request(spec['url'],headers={'User-Agent':'qwen38-recipe-lab'}),timeout=60) as response:
            data=response.read(spec['bytes']+1)
        if not matches(spec,data):
            raise ValueError(f"Source mismatch: {file.name}")
        file.write_bytes(data)
        print('verified',file.name,flush=True)

if __name__=='__main__': main()
