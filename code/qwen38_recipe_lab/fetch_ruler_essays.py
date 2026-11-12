"""Fetch every author-listed essay input with fixed Git revision and bounded IO."""
import argparse,concurrent.futures,hashlib,json,subprocess,threading,time
from datetime import datetime,timedelta
from pathlib import Path
from urllib.request import Request,urlopen


def sha(data):return hashlib.sha256(data).hexdigest()
def get(url,cap):
    with urlopen(Request(url,headers={'User-Agent':'qwen38-local-research'}),timeout=20) as stream:data=stream.read(cap+1)
    assert len(data)<=cap,'Per-input cap exceeded'
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deadline',required=True);args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None
    start=time.monotonic();plan_path=Path('runs/ruler-essay-intake-plan.json');state_path=Path('runs/ruler-essay-intake.json')
    profile=json.loads(Path('profiles/ruler_mac.json').read_text());core=json.loads(Path('ruler_data.lock.json').read_text())
    core_bytes=sum(x['bytes'] for x in core['files']);assert len(core['files'])==3 and core_bytes+30000000<=profile['raw_data_cap_bytes']
    source=Path('.cache/ruler-sources/PaulGrahamEssays_URLs.txt')
    spec=next(s for s in json.loads(Path('ruler_sources.lock.json').read_text()) if s['file']==str(source));assert sha(source.read_bytes())==spec['sha256']
    if plan_path.exists():plan=json.loads(plan_path.read_text());assert plan['source_list_sha256']==spec['sha256']
    else:
        git=json.loads(get('https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/commits/main',1000000))['sha']
        inputs=[]
        for original in source.read_text().splitlines():
            if '.html' in original:
                assert original.startswith('http://www.paulgraham.com/')
                url=original.replace('http://','https://',1);kind='html'
            else:
                old='https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/'
                assert original.startswith(old);url=original.replace(old,f'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/{git}/',1);kind='text'
            inputs.append(dict(original_url=original,url=url,kind=kind,file='data/ruler-source/essays/'+kind+'/'+url.rsplit('/',1)[-1]))
        assert len(inputs)==218 and len({s['file'] for s in inputs})==218
        plan=dict(at=datetime.now().astimezone().isoformat(),source_list_sha256=spec['sha256'],git_revision=git,inputs=inputs,essay_cap_bytes=30000000,global_raw_stored_cap_bytes=profile['raw_data_cap_bytes'],scope='All218 raw inputs; HTTPS for same publisher, pinned Git snapshot, no original private Qwen corpus equivalence claim')
        plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(files={},errors={},started=datetime.now().astimezone().isoformat())
    mutex=threading.Lock()
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    def fetch(item):
        name=item['file'];path=Path(name)
        try:
            if time.monotonic()-start>=1180 or datetime.now().astimezone()>=deadline-timedelta(seconds=180):raise InterruptedError('Time budget')
            if int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))!=1:raise InterruptedError('Native pressure')
            with mutex:existing=state['files'].get(name)
            if existing:
                assert path.stat().st_size==existing['bytes'] and sha(path.read_bytes())==existing['sha256'];return
            assert not path.exists(),'Unrecorded existing input: preserve for review'
            data=get(item['url'],2000000)
            assert data and (b'<html' in data.lower() or item['kind']=='text'),'Unexpected HTML body'
            with mutex:
                total=sum(v['bytes'] for v in state['files'].values())+len(data)
                assert total<=plan['essay_cap_bytes'] and core_bytes+total<=plan['global_raw_stored_cap_bytes'],'Raw corpus cap'
                path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+'.part');tmp.write_bytes(data);tmp.replace(path)
                state['files'][name]=dict(item,bytes=len(data),sha256=sha(data),fetched_at=datetime.now().astimezone().isoformat());state['errors'].pop(name,None);save()
        except Exception as error:
            with mutex:state['errors'][name]=dict(type=type(error).__name__,message=str(error));save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,plan['inputs']))
    state.update(ended=datetime.now().astimezone().isoformat(),status='all_raw_inputs_fetched' if len(state['files'])==218 and not state['errors'] else 'partial_requires_review',full_ruler_prepared=False,scope='Raw IO only; no corpus text conversion, model calls or RULER scores');save()
    print(state['status'],len(state['files']),'of218',len(state['errors']),'errors',flush=True)
    if state['status']!='all_raw_inputs_fetched':raise SystemExit(1)


if __name__=='__main__':main()
