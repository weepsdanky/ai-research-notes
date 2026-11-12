"""Stream losslessly compressed author500 RULER cases under unchanged raw-data cap."""
import argparse,ast,fcntl,gzip,hashlib,json,os,resource,runpy,shutil,signal,subprocess,sys,time,types
from datetime import datetime,timedelta
from pathlib import Path
from ruler_tasks import LocalTokenizer,configurations,digest

ROOT=Path('data/ruler-public500-repair')
PROFILE=Path('profiles/ruler_public500_repair.json')


def pressure():return int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class LimitedWriter:
    def __init__(self,stream,cap):self.stream,self.cap=stream,cap
    def write(self,data):
        if self.stream.tell()+len(data)>self.cap:raise OverflowError('Compressed64MB cap; preserve partial for review')
        return self.stream.write(data)
    def __getattr__(self,name):return getattr(self.stream,name)


def frozen():
    meta=json.loads((ROOT/'meta.json').read_text());assert meta['status']=='complete_frozen_mac_cohort' and meta['total_rows']==13000 and meta['validated_rows']==13000 and len(meta['cases'])==26
    assert all(digest(name)==sha for name,sha in meta['identity']['sources'].items())
    previous=json.loads(Path('data/ruler-mac/meta.json').read_text());assert digest('data/ruler-mac/meta.json')==meta['identity']['original16_meta_sha256']
    assert all(digest(name)==sha for name,sha in previous['identity']['sources'].items())
    assert all(digest(name)==sha for name,sha in previous['identity']['raw'].items())
    assert digest('data/ruler-source/PaulGrahamEssays.json')==previous['identity']['essay_sha256']
    for case in meta['cases'].values():assert digest(case['file'])==case['sha256']
    configurations();return meta


def rows(meta=None,codec=None):
    from tokenizers import Tokenizer
    meta=frozen() if meta is None else meta;codec=Tokenizer.from_file('.cache/references/tokenizer.json') if codec is None else codec;configs=configurations()
    for length in (4096,8192):
        for task,config in configs.items():
            case=meta['cases'][f'{length}/{task}'];hasher=hashlib.sha256();count=0
            with gzip.open(case['file'],'rb') as stream:
                for index,line in enumerate(stream):
                    hasher.update(line);row=json.loads(line);assert index==row['index'] and row['outputs'] and all(isinstance(s,str) and s for s in row['outputs'])
                    prompt=row['input']+row['answer_prefix'];ids=codec.encode(prompt,add_special_tokens=False).ids;assert len(ids)+config['tokens_to_generate']==row['length']<=length
                    if config['task']=='niah':assert all(a in prompt for a in row['outputs'])
                    count+=1;yield dict(source_id=f'{length}/{task}:{index}',task=task,kind=config['task'],context=length,tokens=config['tokens_to_generate'],author_row=row,source_file_sha256=case['uncompressed_sha256'],prompt=prompt,input_ids=ids)
            assert count==500 and hasher.hexdigest()==case['uncompressed_sha256']


def author_child(task,length,destination):
    previous=json.loads(Path('data/ruler-mac/meta.json').read_text());case=previous['cases'][f'{length}/{task}'];proof=json.loads((Path(case['file']).parents[1]/'child.json').read_text());command=list(proof['command']);command[command.index('--num_samples')+1]='500';command[command.index('--save_dir')+1]=str(destination)
    source=Path(command[0]);assert digest(source)==proof['source_sha256'];author=source.parent.parent;sys.path[:0]=[str(source.parent),str(author)]
    shim=types.ModuleType('tokenizer');shim.select_tokenizer=lambda kind,path:LocalTokenizer(path) if kind=='hf' else (_ for _ in ()).throw(ValueError(kind));sys.modules['tokenizer']=shim
    sys.argv=command;repair=None;effective=None
    if task=='qa_2':
        tree=ast.parse(source.read_text());changed=[]
        class Clamp(ast.NodeTransformer):
            def visit_AugAssign(self,node):
                if isinstance(node.target,ast.Name) and node.target.id=='used_docs' and isinstance(node.op,ast.Sub):
                    changed.append(node.lineno)
                    return ast.copy_location(ast.Assign(targets=[node.target],value=ast.Call(func=ast.Name(id='max',ctx=ast.Load()),args=[ast.Name(id='incremental',ctx=ast.Load()),ast.BinOp(left=ast.Name(id='used_docs',ctx=ast.Load()),op=ast.Sub(),right=node.value)],keywords=[])),node)
                return node
        tree=ast.fix_missing_locations(Clamp().visit(tree));assert len(changed)==1
        repair='used_docs=max(incremental,used_docs-incremental)';effective=hashlib.sha256(ast.unparse(tree).encode()).hexdigest()
        exec(compile(tree,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
    else:runpy.run_path(str(source),run_name='__main__')
    save(destination/'child.json',dict(command=command,source_sha256=digest(source),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,repair=repair,effective_ast_sha256=effective))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline');p.add_argument('--resume',action='store_true');p.add_argument('--child',action='store_true');p.add_argument('--task');p.add_argument('--length',type=int);p.add_argument('--destination');a=p.parse_args()
    if a.child:return author_child(a.task,a.length,Path(a.destination))
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo is not None;start=time.monotonic();lock=Path('runs/hf-mmmlu-global.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    profile=json.loads(PROFILE.read_text());previous=json.loads(Path('data/ruler-mac/meta.json').read_text());assert previous['status']=='complete_frozen_mac_cohort' and previous['total_rows']==416
    assert all(digest(name)==sha for name,sha in previous['identity']['sources'].items())
    assert all(digest(name)==sha for name,sha in previous['identity']['raw'].items()) and digest('data/ruler-source/PaulGrahamEssays.json')==previous['identity']['essay_sha256']
    configs=configurations()
    for item in json.loads(Path('ruler_sources.lock.json').read_text()):
        if '/scripts/data/' in item['url'] and not item['url'].endswith('english_words.json'):assert digest(Path('.cache/ruler-author/data')/item['url'].split('/scripts/data/',1)[1])==item['sha256']
    for item in json.loads(Path('ruler_nltk.lock.json').read_text())['files']:assert digest(item['file'])==item['sha256']
    sources=['ruler_public500_repair.py','ruler_public500.py','data/ruler-public500/meta.json','ruler_tasks.py',str(PROFILE),'ruler_sources.lock.json','ruler_data.lock.json','ruler_nltk.lock.json','profiles/ruler-prep-requirements.txt','.cache/references/tokenizer.json']
    identity=dict(sources={name:digest(name) for name in sources},original16_meta_sha256=digest('data/ruler-mac/meta.json'),samples=500,seed=42)
    ROOT.mkdir(parents=True,exist_ok=a.resume);path=ROOT/'meta.json'
    if a.resume:
        meta=json.loads(path.read_text());assert meta['identity']==identity
        if meta['status']=='complete_frozen_mac_cohort':print('Already complete; preserve frozen metadata');return
    else:assert not path.exists();meta=dict(started=datetime.now().astimezone().isoformat(),identity=identity,status='preparing',cases=json.loads(Path('data/ruler-public500/meta.json').read_text())['cases'],attempts=[],variant='explicit QA retry-clamp Mac repair; original500 remains failed, no question/doc/length/budget change')
    save(path,meta);env=dict(os.environ,NLTK_DATA=str(Path('.cache/ruler-nltk').resolve()),PYTHONHASHSEED='0')
    for length in profile['lengths']:
        for task,config in configs.items():
            key=f'{length}/{task}'
            if key in meta['cases']:assert digest(meta['cases'][key]['file'])==meta['cases'][key]['sha256'];continue
            if pressure()!=1 or time.monotonic()-start>=1180 or datetime.now().astimezone()>=deadline-timedelta(seconds=200):meta['status']='resource_or_time_stop';save(path,meta);return
            attempt=ROOT/'attempts'/f'{length}-{task}-{len(meta["attempts"]):03d}';attempt.mkdir(parents=True,exist_ok=False);item=dict(key=key,directory=str(attempt),started=datetime.now().astimezone().isoformat());meta['attempts'].append(item);save(path,meta)
            with (attempt/'generator.log').open('x') as log,(attempt/'resources.jsonl').open('x') as samples:
                process=subprocess.Popen([sys.executable,'ruler_public500_repair.py','--child','--task',task,'--length',str(length),'--destination',str(attempt.resolve())],stdout=log,stderr=subprocess.STDOUT,env=env);item['pid']=process.pid;save(path,meta);stopping=None
                while process.poll() is None:
                    level=pressure();probe=subprocess.run(['ps','-p',str(process.pid),'-o','rss='],capture_output=True,text=True).stdout.strip();rss=int(probe or 0)*1024;scratch=sum(f.stat().st_size for f in ROOT.rglob('validation.jsonl'));now=datetime.now().astimezone();samples.write(json.dumps(dict(at=now.isoformat(),pid=process.pid,pressure=level,rss_bytes=rss,scratch_bytes=scratch))+'\n');samples.flush()
                    if stopping is None and (level>=2 or rss>profile['preparation_peak_rss_cap_bytes'] or scratch>profile['scratch_cap_bytes'] or time.monotonic()-start>=1180 or now>=deadline-timedelta(seconds=180)):
                        stopping=time.monotonic();item['stop_request']=dict(at=now.isoformat(),pressure=level,rss_bytes=rss,scratch_bytes=scratch);process.send_signal(signal.SIGINT)
                    elif stopping is not None and time.monotonic()-stopping>=10:process.terminate()
                    time.sleep(.5)
            item.update(returncode=process.returncode,ended=datetime.now().astimezone().isoformat());save(path,meta)
            if process.returncode or stopping:meta['status']='generator_failure_or_resource_stop';save(path,meta);return
            raw=attempt/task/'validation.jsonl';assert raw.stat().st_size<=profile['scratch_cap_bytes'];old=Path(previous['cases'][key]['file']).read_bytes()
            with raw.open('rb') as stream:assert stream.read(len(old))==old,'Author500 must retain exact original16 prefix'
            proof=json.loads((attempt/'child.json').read_text());assert proof['peak_rss_bytes']<=profile['preparation_peak_rss_cap_bytes']
            target=ROOT/f'{length}-{task}.jsonl.gz';assert not target.exists();uncompressed=digest(raw)
            try:
                with raw.open('rb') as src,target.open('xb') as output:
                    with gzip.GzipFile(filename='',fileobj=LimitedWriter(output,profile['prepared_cap_bytes']-sum(c['compressed_bytes'] for c in meta['cases'].values())),mode='wb',mtime=0,compresslevel=9) as stream:shutil.copyfileobj(src,stream,1024*1024)
            except OverflowError:
                meta['status']='compressed_budget_exceeded';save(path,meta);raise
            stored=sum(Path(c['file']).stat().st_size for c in meta['cases'].values())+target.stat().st_size
            if stored>profile['prepared_cap_bytes']:
                item['compressed_cap_failure']=dict(stored=stored);meta['status']='compressed_budget_exceeded';save(path,meta);return
            case=dict(file=str(target),sha256=digest(target),uncompressed_sha256=uncompressed,uncompressed_bytes=raw.stat().st_size,compressed_bytes=target.stat().st_size,rows=500,length=length,task=task,kind=config['task'],tokens=config['tokens_to_generate'],first16_byte_prefix_identical=True,author_proof=str(attempt/'child.json'),peak_rss_bytes=proof['peak_rss_bytes']);meta['cases'][key]=case;raw.unlink();save(path,meta);print(key,'accepted500',case['uncompressed_bytes'],case['compressed_bytes'],flush=True)
    meta.update(status='source_generation_complete_pending_validation',total_rows=13000,ended=datetime.now().astimezone().isoformat(),full_paper_protocol=False,compressed_total_bytes=sum(c['compressed_bytes'] for c in meta['cases'].values()),uncompressed_total_bytes=sum(c['uncompressed_bytes'] for c in meta['cases'].values()));save(path,meta)
    count=0
    for row in rows(meta):
        count+=1
        if count%500==0:assert time.monotonic()-start<1180 and pressure()==1 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=profile['preparation_peak_rss_cap_bytes'] and datetime.now().astimezone()<deadline-timedelta(seconds=180)
    assert count==13000;meta.update(status='complete_frozen_mac_cohort',validated_rows=13000,ended=datetime.now().astimezone().isoformat());save(path,meta);print('All13000 rows and token budgets verified',flush=True)


if __name__=='__main__':main()
