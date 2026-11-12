"""Fetch complete public MMLU/BBH/MGSM evaluation sets, never training corpora.

All upstream revisions are immutable. Individual files are capped at 8MB and
this suite at 64MB; hashes become part of the normal evaluation source lock.
"""
import concurrent.futures,hashlib,json,time,threading
from pathlib import Path
from urllib.request import urlopen

ROOT=Path('data/eval-source')
MMLU='c30699e8356da336a370243923dbaf21066bb9fe'
BBH='9ee07bd481feebf959a6b59d61ea57bdcf30964d'
MGSM='452a21ad3dae5668c06ceeac21ff073e1e40f9be'


def read(url,cap):
    for attempt in range(3):
        try:
            with urlopen(url,timeout=40) as response:raw=response.read(cap+1)
            if len(raw)>cap:raise ValueError(f'Byte cap exceeded: {url}')
            return raw
        except Exception:
            if attempt==2:raise
            time.sleep(.5*2**attempt)


def main():
    lock=Path('eval_sources.lock.json');existing=json.loads(lock.read_text())
    by_name={s['name']:s for s in existing};ROOT.mkdir(parents=True,exist_ok=True)
    mmlu=json.loads(read(f'https://huggingface.co/api/datasets/cais/mmlu/revision/{MMLU}',4_000_000))
    subjects=sorted({s['rfilename'].split('/')[0] for s in mmlu['siblings'] if '/test-' in s['rfilename'] and not s['rfilename'].startswith('all/')})
    assert len(subjects)==57
    tree=json.loads(read(f'https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/git/trees/{BBH}?recursive=1',4_000_000))
    tasks=sorted(s['path'][4:-5] for s in tree['tree'] if s['path'].startswith('bbh/') and s['path'].endswith('.json'))
    assert len(tasks)==27
    languages=['bn','de','en','es','fr','ja','ru','sw','te','th','zh']
    plan=[]
    consumed=0;mutex=threading.Lock()
    def reserve(size):
        nonlocal consumed
        with mutex:
            if consumed+size>64_000_000:raise ValueError('Suite byte cap exceeded before persisting another file')
            consumed+=size
    def add(name,repo,rev,path,hf=False):
        base=f'https://huggingface.co/datasets/{repo}/resolve/{rev}' if hf else f'https://raw.githubusercontent.com/{repo}/{rev}'
        plan.append(dict(name=name,repo=repo,revision=rev,url=base+'/'+path))
    for subject in subjects:
        for split in ('dev','test'):
            add(f'mmlu-{subject}-{split}.parquet','cais/mmlu',MMLU,f'{subject}/{split}-00000-of-00001.parquet',True)
    for task in tasks:
        for directory,ext in (('bbh','json'),('cot-prompts','txt')):
            add(f'bbh-{task}.{ext}','suzgunmirac/BIG-Bench-Hard',BBH,f'{directory}/{task}.{ext}')
    for lang in languages:
        add(f'mgsm-{lang}.tsv','google-research/url-nlp',MGSM,f'mgsm/mgsm_{lang}.tsv')
    def fetch(spec):
        path=ROOT/spec['name'];old=by_name.get(spec['name'])
        if old and path.exists():
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest==old['sha256']:
                reserve(old['bytes']);return old
        raw=read(spec['url'],8_000_000);digest=hashlib.sha256(raw).hexdigest()
        if old and digest!=old['sha256']:raise ValueError(f'Locked source drift: {spec["name"]}')
        reserve(len(raw))
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(raw);temp.replace(path)
        return dict(spec,sha256=digest,bytes=len(raw))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fetched=list(pool.map(fetch,plan))
    assert sum(s['bytes'] for s in fetched)<=64_000_000
    for spec in fetched:by_name[spec['name']]=spec
    lock.write_text(json.dumps(list(by_name.values()),indent=2)+'\n')
    suite=dict(mmlu_subjects=subjects,bbh_subtasks=tasks,mgsm_languages=languages,
        status='complete dataset coverage for these three families; evaluation not yet complete',
        bbh_note='27 published subtask configurations, grouped into 23 task families; paper aggregation unspecified',
        byte_cap=64_000_000,actual_bytes=sum(s['bytes'] for s in fetched))
    Path('eval_suites.json').write_text(json.dumps(suite,indent=2)+'\n')
    print(json.dumps(suite,indent=2),flush=True)


if __name__=='__main__':main()
