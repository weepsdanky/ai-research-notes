"""Read-only validation in a separate guarded CPU process, without optimizer RAM.

The numerical evaluator/data batches match training. mmap+assign loads model
weights only; parent checkpoint/source identities and partial metrics are bound.
"""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta


def load(path):return json.loads(Path(path).read_text())
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def native(command):return subprocess.check_output(command,text=True)
def pressure():return int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def worker(a):
    import torch
    from qwen35_controls import ControlConfig,Qwen35Control
    from train_qwen35_controls import Corpus,evaluate
    assert os.getppid()==int(os.environ['QWEN35_VALIDATION_GUARD_PID']);torch.set_num_threads(1)
    profile=load(a.profile);audit=load(a.audit);cp_sha=digest(a.checkpoint)
    assert audit['status']=='passed_checkpoint_checks' and audit['checkpoint_sha256']==cp_sha and audit['profile_sha256']==digest(a.profile)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    assert saved['profile']==profile and all(digest(p)==sha for p,sha in saved['source_hashes'].items())
    identity=dict(checkpoint_sha256=cp_sha,profile_sha256=digest(a.profile),audit_sha256=digest(a.audit),validator_sha256=digest(__file__),source_hashes=saved['source_hashes'],step=saved['step'],phase=saved.get('phase','4k_short'),tokens=saved['tokens'],query_chunk=a.query_chunk,execution='CPU no-grad; mmap+assign model only; explicit query tiling, unchanged full context/softmax/evaluator/data')
    if a.resume:
        result=load(a.out);assert result['identity']==identity and result['status']!='complete'
    else:assert not Path(a.out).exists();result=dict(status='running',identity=identity,results={})
    # Construction on meta retains public parameter names without a second full copy.
    runtime_config=dict(saved['config'],query_chunk=a.query_chunk)
    with torch.device('meta'):model=Qwen35Control(ControlConfig(**runtime_config),saved['arm'],seed=profile['seed'])
    model.load_state_dict(saved['model'],strict=True,assign=True);del saved;model.eval()
    corpora=[Corpus(row['directory'],row['meta_sha256'],32768) for row in profile['data']]
    write(a.out,result)
    for seq,label in [(32768,'val32k'),(4096,'val4k')]:
        for index,corpus in enumerate(corpora):
            key=f'{label}/val{index}/lm'
            if key in result['results']:continue
            metric=evaluate(model,[corpus],dict(profile,seq=seq),'cpu')['val0/lm']
            result['results'][key]=metric;write(a.out,result);print(key,metric,flush=True)
    assert digest(a.checkpoint)==cp_sha
    result.update(status='complete',completed_at=datetime.now().astimezone().isoformat());write(a.out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);p.add_argument('--prefix',required=True);p.add_argument('--deadline',required=True);p.add_argument('--query-chunk',type=int,default=16);p.add_argument('--resume',action='store_true');p.add_argument('--prior-state');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();assert a.query_chunk>0
    if a.worker:return worker(a)
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo and datetime.now().astimezone()+timedelta(seconds=660)<deadline<=datetime.now().astimezone()+timedelta(hours=2)
    prefix=Path(a.prefix);state_path=Path(str(prefix)+'-state.json');assert not any(Path(str(prefix)+s).exists() for s in ['-state.json','.log','-resources.jsonl'])
    state=dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),status='preflight',signals=[],source_sha256=digest(__file__),checkpoint_sha256=digest(a.checkpoint),profile_sha256=digest(a.profile),audit_sha256=digest(a.audit),goal_status='active');write(state_path,state);child=None;handlers={};requests=[]
    try:
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]);assert pressure()==1 and free>=55
            assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
            if a.resume:
                assert a.prior_state;prior=load(a.prior_state)
                for pid in [prior['pid'],prior['child_pid']]:assert not subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True).stdout.strip()
                assert prior['source_sha256']==state['source_sha256'] and prior['checkpoint_sha256']==state['checkpoint_sha256'] and prior['profile_sha256']==state['profile_sha256'] and digest(a.out)==prior['output_sha256']
                assert not prior['signals'] and prior['returncode']==0,'Resource/unknown validation stop requires fresh review, not ordinary resume'
            else:assert not Path(a.out).exists()
            command=[sys.executable,__file__,*sys.argv[1:],'--worker'];env={**os.environ,'QWEN35_VALIDATION_GUARD_PID':str(os.getpid())}
            for signum in [signal.SIGINT,signal.SIGTERM]:handlers[signum]=signal.signal(signum,lambda n,f:requests.append(n))
            with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as samples:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);state.update(status='running',child_pid=child.pid,command=command,initial_pressure=1,initial_free_percent=free);write(state_path,state);tick=time.monotonic();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure();ps=subprocess.run(['ps','-p',str(child.pid),'-o','pid=,ppid=,rss=,command='],capture_output=True,text=True).stdout
                    samples.write(json.dumps(dict(at=now.isoformat(),pressure=level,child=ps,swap=native(['sysctl','vm.swapusage'])))+'\n');samples.flush()
                    reason='memory_pressure' if level>=2 else 'parent_signal' if requests else 'wall_budget' if time.monotonic()-tick>=1180 else 'deadline' if now>=deadline-timedelta(seconds=600) else None
                    if reason and stop is None:child.send_signal(signal.SIGINT);stop=time.monotonic();state['signals'].append(dict(signal='SIGINT',reason=reason,pressure=level));write(state_path,state)
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):child.kill();state['signals'].append(dict(signal='SIGKILL',reason='own_child_grace_exceeded'));break
                    time.sleep(2)
                child.wait();state.update(status='terminal',returncode=child.returncode,ended=datetime.now().astimezone().isoformat())
            if Path(a.out).exists():state.update(output=a.out,output_sha256=digest(a.out),metrics_completed=len(load(a.out)['results']))
            assert digest(a.checkpoint)==state['checkpoint_sha256'],'Read-only validation changed checkpoint'
            state['status']='complete' if child.returncode==0 and not state['signals'] and load(a.out)['status']=='complete' else 'stop_requires_review';write(state_path,state)
    except BaseException as e:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='needs_review',error_type=type(e).__name__,error=str(e));write(state_path,state);raise
    finally:
        for n,h in handlers.items():signal.signal(n,h)


if __name__=='__main__':main()
