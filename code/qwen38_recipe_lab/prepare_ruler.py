"""Serial resource-guarded execution of all13 locked RULER author generators."""
import argparse,fcntl,hashlib,json,os,resource,runpy,shutil,signal,subprocess,sys,time,types
from datetime import datetime,timedelta
from pathlib import Path
from ruler_tasks import LocalTokenizer,configurations,digest


def pressure():return int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def child(task,length,destination):
    configs=configurations();config=configs[task];root=Path.cwd();author=root/'.cache/ruler-author/data';sys.path[:0]=[str(author/'synthetic'),str(author)]
    shim=types.ModuleType('tokenizer')
    def select(kind,path):
        assert kind=='hf' and Path(path).resolve()==root/'.cache/references/tokenizer.json'
        return LocalTokenizer(path)
    shim.select_tokenizer=select;sys.modules['tokenizer']=shim
    argv=['--save_dir',str(destination),'--save_name',task,'--subset','validation','--tokenizer_path',str(root/'.cache/references/tokenizer.json'),'--tokenizer_type','hf','--max_seq_length',str(length),'--tokens_to_generate',str(config['tokens_to_generate']),'--num_samples','16','--random_seed','42','--template',config['template']]
    for key,value in config['args'].items():argv += ['--'+key,str(value)]
    source=author/'synthetic'/f"{config['task']}.py";sys.argv=[str(source)]+argv
    runpy.run_path(str(source),run_name='__main__')
    write(destination/'child.json',dict(command=sys.argv,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=digest(source)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deadline');parser.add_argument('--resume',action='store_true');parser.add_argument('--child',action='store_true');parser.add_argument('--task');parser.add_argument('--length',type=int);parser.add_argument('--destination');args=parser.parse_args()
    if args.child:return child(args.task,args.length,Path(args.destination))
    deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None
    lock=Path('runs/hf-mmmlu-global.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();root=Path.cwd();out=root/'data/ruler-mac';profile=json.loads(Path('profiles/ruler_mac.json').read_text());configs=configurations();codec=LocalTokenizer('.cache/references/tokenizer.json').codec
    conversion=json.loads(Path('runs/ruler-essay-conversion.json').read_text());assert conversion['status']=='all218_author_conversion_complete' and digest(conversion['file'])==conversion['sha256']
    raw=json.loads(Path('ruler_data.lock.json').read_text())['files']
    for item in raw:assert digest(item['file'])==item['sha256']
    nlplock=json.loads(Path('ruler_nltk.lock.json').read_text())
    for item in nlplock['files']:assert digest(item['file'])==item['sha256']
    sources=['prepare_ruler.py','ruler_tasks.py','profiles/ruler_mac.json','profiles/ruler-prep-requirements.txt','ruler_sources.lock.json','ruler_data.lock.json','ruler_nltk.lock.json','.cache/references/tokenizer.json']
    identity=dict(sources={name:digest(name) for name in sources},essay_sha256=conversion['sha256'],raw={item['file']:item['sha256'] for item in raw},configurations=list(configs),lengths=profile['lengths'],samples=16,seed=42,tokenizer_adapter='locked tokenizers backend encode.tokens/decoder.decode; no special tokens',pythonhashseed='0')
    out.mkdir(parents=True,exist_ok=args.resume);meta_path=out/'meta.json'
    if args.resume:meta=json.loads(meta_path.read_text());assert meta['identity']==identity,'Source drift; preserve prior corpus'
    else:assert not meta_path.exists();meta=dict(started=datetime.now().astimezone().isoformat(),identity=identity,cases={},attempts=[],status='preparing')
    author=root/'.cache/ruler-author/data';(author/'synthetic/json').mkdir(parents=True,exist_ok=True)
    for item in json.loads(Path('ruler_sources.lock.json').read_text()):
        marker='/scripts/data/'
        if marker not in item['url']:continue
        relative=item['url'].split(marker,1)[1]
        if relative.endswith('english_words.json'):continue
        target=author/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==item['sha256']
        else:shutil.copyfile(item['file'],target)
    for item in raw+[conversion]:
        target=author/'synthetic/json'/Path(item['file']).name
        if target.exists():assert target.resolve()==(root/item['file']).resolve()
        else:target.symlink_to(root/item['file'])
    env=dict(os.environ,NLTK_DATA=str(root/'.cache/ruler-nltk'),PYTHONHASHSEED='0')
    write(meta_path,meta)
    for length in profile['lengths']:
        for task,config in configs.items():
            key=f'{length}/{task}'
            if key in meta['cases']:
                item=meta['cases'][key];assert digest(item['file'])==item['sha256'];continue
            if pressure()!=1 or datetime.now().astimezone()>=deadline-timedelta(seconds=200) or time.monotonic()-start>=1180:
                meta.update(status='resource_or_time_stop',ended=datetime.now().astimezone().isoformat());write(meta_path,meta);return
            attempt=out/'attempts'/f'{length}-{task}-{len(meta["attempts"]):03d}';attempt.mkdir(parents=True,exist_ok=False)
            item=dict(key=key,started=datetime.now().astimezone().isoformat(),directory=str(attempt.relative_to(root)));meta['attempts'].append(item);write(meta_path,meta)
            with (attempt/'generator.log').open('x') as log,(attempt/'resources.jsonl').open('x') as samples:
                command=[sys.executable,'prepare_ruler.py','--child','--task',task,'--length',str(length),'--destination',str(attempt)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);tick=time.monotonic();stopping=None
                while process.poll() is None:
                    level=pressure();now=datetime.now().astimezone();probe=subprocess.run(['ps','-p',str(process.pid),'-o','rss='],capture_output=True,text=True).stdout.strip();rss=int(probe or 0)*1024
                    samples.write(json.dumps(dict(at=now.isoformat(),pressure=level,rss_bytes=rss))+'\n');samples.flush()
                    if stopping is None and (level>=2 or rss>profile['preparation_peak_rss_cap_bytes'] or time.monotonic()-tick>=1100 or time.monotonic()-start>=1180 or now>=deadline-timedelta(seconds=180)):
                        stopping=time.monotonic();item['stop_request']=dict(at=now.isoformat(),pressure=level,rss_bytes=rss);process.send_signal(signal.SIGINT)
                    elif stopping and time.monotonic()-stopping>=10:process.terminate()
                    time.sleep(.5)
            item.update(ended=datetime.now().astimezone().isoformat(),returncode=process.returncode);write(meta_path,meta)
            if process.returncode or stopping:
                meta['status']='generator_failure_or_resource_stop';write(meta_path,meta);return
            generated=attempt/task/'validation.jsonl';rows=[json.loads(line) for line in generated.read_text().splitlines()];assert len(rows)==16 and [r['index'] for r in rows]==list(range(16))
            for row in rows:
                ids=codec.encode(row['input']+row['answer_prefix'],add_special_tokens=False).ids
                assert len(ids)+config['tokens_to_generate']==row['length']<=length
                assert row['outputs'] and all(isinstance(s,str) and s for s in row['outputs'])
            peak=json.loads((attempt/'child.json').read_text())['peak_rss_bytes'];assert peak<=profile['preparation_peak_rss_cap_bytes']
            assert sum(Path(x['file']).stat().st_size for x in meta['cases'].values())+generated.stat().st_size<=profile['prepared_cap_bytes']
            meta['cases'][key]=dict(file=str(generated.relative_to(root)),sha256=digest(generated),rows=16,length=length,task=task,kind=config['task'],tokens=config['tokens_to_generate'],peak_rss_bytes=peak);write(meta_path,meta);print(key,'accepted16',flush=True)
    meta.update(status='complete_frozen_mac_cohort',total_rows=sum(x['rows'] for x in meta['cases'].values()),ended=datetime.now().astimezone().isoformat(),full_paper_protocol=False)
    assert len(meta['cases'])==26 and meta['total_rows']==416;write(meta_path,meta)


if __name__=='__main__':main()
