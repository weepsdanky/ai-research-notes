"""Retokenize the existing bounded, hash-verified proxy data with official tokenizer.

No new OpenWebText download. Source documents retain their original train/val split.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    lock=json.loads(Path('sources.lock.json').read_text())
    spec=next(s for s in lock if s['name']=='tokenizer.json')
    tokenizer_path=Path('.cache/references/tokenizer.json')
    assert sha(tokenizer_path)==spec['sha256']
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    source=json.loads((args.source/'meta.json').read_text())
    if source['tokenizer']!='utf8-bytes+eod256': raise ValueError('Unexpected source format')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(tokenizer='Qwen/Qwen3.8-Flash-Next',tokenizer_revision=spec['revision'],
                tokenizer_sha256=spec['sha256'],tokenizer_actual_vocab=tokenizer.get_vocab_size(),
                model_vocab_size=248320,eos=248044,dtype='<u4',source_meta=source,files={})
    for split in ('train','val'):
        file=args.source/f'{split}.bin'
        if sha(file)!=source['files'][split]['sha256']: raise ValueError('Source data hash mismatch')
        data=np.memmap(file,mode='r',dtype='<u2')
        boundaries=np.flatnonzero(data==256)
        start=0; count=0; docs=0; trimmed=0
        out=args.out/f'{split}.bin'
        with out.open('wb') as f:
            for end in boundaries:
                raw=data[start:end].astype('u1').tobytes(); start=int(end)+1
                try: text=raw.decode('utf-8')
                except UnicodeDecodeError as exc:
                    if exc.end!=len(raw) or exc.reason!='unexpected end of data': raise
                    trimmed+=len(raw)-exc.start; text=raw[:exc.start].decode('utf-8')
                ids=tokenizer.encode(text,add_special_tokens=False).ids+[248044]
                if max(ids)>=248320: raise ValueError('Tokenizer / model vocab mismatch')
                np.asarray(ids,dtype='<u4').tofile(f); count+=len(ids); docs+=1
        if start!=len(data): raise ValueError('Source missing final EOD')
        report['files'][split]=dict(tokens=count,documents=docs,sha256=sha(out),trimmed_incomplete_utf8_bytes=trimmed)
    (args.out/'meta.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
