"""Check cached inference against full-prefix logits of real trained controls.

This is numerical integration, not a quality benchmark or speed measurement.
Each manifest entry binds one checkpoint and its independent training audit.
"""
import argparse,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta
from run_dense_text_guarded import load,digest,native,pressure,absent


def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def worker(a):
    import torch
    from tokenizers import Tokenizer
    from qwen35_controls import ControlConfig,Qwen35Control
    from qwen35_inference import ControlInference
    assert os.getppid()==int(os.environ['QWEN35_CACHE_AUDIT_GUARD_PID'])
    torch.set_num_threads(1);entries=load(a.manifest)
    spec=load('qwen35_tokenizer_sources.lock.json')[0];assert digest(spec['file'])==spec['sha256']
    tokenizer=Tokenizer.from_file(spec['file'])
    text=('ROMEO: The world is full of words.\nScience explains observations with evidence.\n'*100)
    pool=tokenizer.encode(text,add_special_tokens=False).ids;assert len(pool)>259
    result=dict(status='running',manifest_sha256=digest(a.manifest),source_sha256=digest(__file__),kernel_sha256=digest('qwen35_inference.py'),tokenizer_sha256=spec['sha256'],checks=[],scope='CPU trained-weight numerical/cache integration only; mixed stages recorded explicitly, no benchmark accuracy or kernel-speed claim')
    save(a.out,result)
    for entry in entries:
        audit=load(entry['audit']);cp=entry['checkpoint'];cp_sha=digest(cp)
        assert audit['status']=='passed_checkpoint_checks' and audit['checkpoint_sha256']==cp_sha
        saved=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
        assert audit['profile_sha256']==saved['profile_sha256'] and all(digest(p)==sha for p,sha in saved['source_hashes'].items())
        with torch.device('meta'):model=Qwen35Control(ControlConfig(**saved['config']),saved['arm'],seed=saved['profile']['seed'])
        model.load_state_dict(saved['model'],strict=True,assign=True);model.eval()
        record=dict(arm=saved['arm'],phase=saved.get('phase','4k_short'),step=saved['step'],tokens=saved['tokens'],checkpoint=cp,checkpoint_sha256=cp_sha,audit_sha256=digest(entry['audit']),config=saved['config'],max_abs_logit_error=0.,greedy_ids=[],ragged_decode_steps=16,single_decode_steps=16);del saved
        engine=ControlInference(model,query_chunk=16,max_context=512)
        def compare(actual,expected):
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
            assert torch.equal(actual.argmax(-1),expected.argmax(-1)),'Greedy IDs differ'
            record['max_abs_logit_error']=max(record['max_abs_logit_error'],float((actual-expected).abs().max()))
        def teacher(ids):
            with torch.no_grad():hidden,_=model.hidden(ids);return model.head(hidden[:,-1])
        prompts=[torch.tensor([pool[:n]],dtype=torch.long) for n in [3,131,259]]
        caches=[]
        for ids in prompts:
            logits,cache=engine.prefill(ids,chunk=31);compare(logits,teacher(ids));caches.append(cache)
        snapshot=[{k:v.clone() for k,v in layer.items()} for layer in caches[0]['layers']]
        lengths=caches[0]['lengths'].clone();branch=engine.fork(caches[0]);engine.forward(torch.tensor([[pool[7]]]),branch)
        assert torch.equal(caches[0]['lengths'],lengths)
        for layer,before in zip(caches[0]['layers'],snapshot,strict=True):
            for key in before:assert torch.equal(layer[key],before[key]),'Fork mutated prefix'
        single=engine.fork(caches[-1]);ids=prompts[-1].clone();actual=teacher(ids)
        for _ in range(16):
            expected=teacher(ids);compare(actual,expected);token=expected.argmax(-1)[:,None];record['greedy_ids'].append(int(token));ids=torch.cat((ids,token),-1);actual=engine.forward(token,single)
        compare(actual,teacher(ids));packed=engine.pack(caches)
        for _ in range(16):
            expected=torch.cat([teacher(ids) for ids in prompts]);tokens=expected.argmax(-1)[:,None];actual=engine.forward(tokens,packed)
            prompts=[torch.cat((ids,tokens[row:row+1]),-1) for row,ids in enumerate(prompts)]
            compare(actual,torch.cat([teacher(ids) for ids in prompts]))
        if record['arm']=='swa':
            for layer,state in zip(model.layers,packed['layers'],strict=True):
                if getattr(layer.mixer,'window',None) is not None:assert state['k'].shape[-1]==127
        assert digest(cp)==cp_sha;record['status']='passed';result['checks'].append(record);save(a.out,result)
        del model,engine,caches,packed,branch,single,actual
    result.update(status='passed',ended=datetime.now().astimezone().isoformat());save(a.out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--prefix',required=True);p.add_argument('--deadline',required=True);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:return worker(a)
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo and datetime.now().astimezone()+timedelta(seconds=660)<deadline
    assert not Path(a.out).exists();prefix=Path(a.prefix);assert not any(Path(str(prefix)+suffix).exists() for suffix in ['-state.json','.log','-resources.jsonl'])
    path=str(prefix)+'-state.json';state=dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),status='preflight',signals=[],goal_status='active',source_sha256=digest(__file__),manifest_sha256=digest(a.manifest));save(path,state);child=None
    try:
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert pressure()==1 and int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1])>=55
            assert all(os.environ.get(k) is None for k in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
            command=[sys.executable,__file__,'--manifest',a.manifest,'--out',a.out,'--prefix',a.prefix,'--deadline',a.deadline,'--worker'];state['command']=command
            with open(str(prefix)+'.log','x') as log,open(str(prefix)+'-resources.jsonl','x') as resources:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'QWEN35_CACHE_AUDIT_GUARD_PID':str(os.getpid())});state.update(status='running',child_pid=child.pid);save(path,state);tick=time.monotonic();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure();rss=subprocess.run(['ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True).stdout
                    resources.write(json.dumps(dict(at=now.isoformat(),pressure=level,rss_kib=rss.strip(),swap=native(['sysctl','vm.swapusage'])))+'\n');resources.flush()
                    reason='memory_pressure' if level>=2 else 'deadline' if now>=deadline-timedelta(seconds=600) else 'wall_budget' if time.monotonic()-tick>=1180 else None
                    if stop is None and reason:child.send_signal(signal.SIGINT);stop=time.monotonic();state['signals'].append(dict(reason=reason,signal='SIGINT',pressure=level));save(path,state)
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):child.kill();state['signals'].append(dict(reason='own_child_grace_exceeded',signal='SIGKILL'));break
                    time.sleep(2)
                child.wait();absent(child.pid)
            state.update(returncode=child.returncode,status='passed' if child.returncode==0 and not state['signals'] and load(a.out)['status']=='passed' else 'stop_requires_review',ended=datetime.now().astimezone().isoformat());save(path,state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='needs_review',error=repr(error));save(path,state);raise


if __name__=='__main__':main()
