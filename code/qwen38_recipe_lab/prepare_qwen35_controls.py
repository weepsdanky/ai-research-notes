"""Retokenize only the existing bounded byte/EOD proxy into a new Qwen3.5 directory."""
import argparse,ctypes,fcntl,hashlib,json,re,subprocess,time
from datetime import datetime
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pressure():
    libc=ctypes.CDLL(None,use_errno=True);value=ctypes.c_int();size=ctypes.c_size_t(ctypes.sizeof(value));call=libc.sysctlbyname
    call.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t];call.restype=ctypes.c_int
    if call(b'kern.memorystatus_vm_pressure_level',ctypes.byref(value),ctypes.byref(size),None,0):raise OSError(ctypes.get_errno(),'Native memory pressure unavailable')
    return value.value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',required=True);a=p.parse_args();deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo is not None and datetime.now().astimezone()<deadline
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert pressure()==1
        free=int(re.search(r'free percentage:\s*(\d+)%',subprocess.check_output(['memory_pressure','-Q'],text=True))[1]);assert free>=55
        spec=json.loads(Path('qwen35_tokenizer_sources.lock.json').read_text())[0];tokenizer_path=Path(spec['file']);assert sha(tokenizer_path)==spec['sha256'];tokenizer=Tokenizer.from_file(str(tokenizer_path));assert tokenizer.id_to_token(248044)=='<|endoftext|>'
        source=json.loads((a.source/'meta.json').read_text());assert source['tokenizer']=='utf8-bytes+eod256'
        assert source['files']['train']['tokens']<=20010000 and source['files']['val']['tokens']<=1010000,'Keep the existing bounded raw subset'
        for split in ['train','val']:assert sha(a.source/f'{split}.bin')==source['files'][split]['sha256']
        a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();report=dict(tokenizer=spec['model'],tokenizer_revision=spec['revision'],tokenizer_sha256=spec['sha256'],tokenizer_actual_vocab=tokenizer.get_vocab_size(),model_vocab_size=248320,eos=248044,dtype='<u4',source_meta=source,source_meta_sha256=sha(a.source/'meta.json'),source_script_sha256=sha(__file__),document_separator='Mac proxy choice: append published eos248044, not a claim about private pretraining separators',files={})
        for split in ['train','val']:
            data=np.memmap(a.source/f'{split}.bin',mode='r',dtype='<u2');assert len(data)==source['files'][split]['tokens'];boundaries=np.flatnonzero(data==256);offset=0;count=0;docs=0;trimmed=0;out=a.out/f'{split}.bin'
            with out.open('xb') as stream:
                for end in boundaries:
                    assert pressure()<2 and datetime.now().astimezone()<deadline and time.monotonic()-started<180,'Resource/time stop: preserve partial new directory for review'
                    raw=data[offset:end].astype('u1').tobytes();offset=int(end)+1
                    try:text=raw.decode('utf-8')
                    except UnicodeDecodeError as error:
                        if error.end!=len(raw) or error.reason!='unexpected end of data':raise
                        trimmed+=len(raw)-error.start;text=raw[:error.start].decode('utf-8')
                    ids=tokenizer.encode(text,add_special_tokens=False).ids+[248044];assert max(ids)<248320;np.asarray(ids,dtype='<u4').tofile(stream);count+=len(ids);docs+=1
            assert offset==len(data);report['files'][split]=dict(tokens=count,documents=docs,sha256=sha(out),trimmed_incomplete_utf8_bytes=trimmed);assert out.stat().st_size==count*4
        report['seconds']=time.monotonic()-started;(a.out/'meta.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(output=str(a.out),files=report['files'],seconds=report['seconds'])))


if __name__=='__main__':main()
