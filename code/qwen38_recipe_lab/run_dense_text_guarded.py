"""Run one full-profile Dense text evaluation call with native resource protection.

No automatic resource recovery. Ordinary resume requires an independently audited
time stop; resource resume needs a fresh one-use SHA-bound review. Code execution
and judging use separate harnesses.
"""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from datetime import datetime,timedelta
from pathlib import Path


def load(path):return json.loads(Path(path).read_text())
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def native(command):return subprocess.check_output(command,text=True)
def pressure():return int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
def absent(pid):assert not subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True).stdout.strip(),f'Prior PID still live: {pid}'


def validate_resource_review(path,prior_state,entry,profile_path,profile,binding,deadline):
    review=load(path);now=datetime.now().astimezone()
    assert Path(review['state_file']).resolve()==Path(prior_state).resolve() and digest(prior_state)==review['state_sha256']
    expected=dict(output=entry['output'],predictions_sha256=entry['predictions_sha256'],meta_sha256=entry['meta_sha256'],audit_file=entry['audit_file'],audit_sha256=entry['audit_sha256'],profile_sha256=digest(profile_path),engine_sha256=profile['engine_sha256'],kernel_sha256=profile['kernel_sha256'],checkpoint_sha256=binding['checkpoint_sha256'],guard_source_sha256=digest(__file__),deadline=deadline.isoformat())
    assert all(review[key]==value for key,value in expected.items()),'Review identity mismatch'
    assert digest(review['samples_file'])==review['samples_sha256']
    samples=load(review['samples_file']);assert len(samples)==12 and all(r['pressure']==1 and r['free']>=55 for r in samples)
    stamps=[datetime.fromisoformat(r['at']) for r in samples]
    assert all(t.tzinfo is not None for t in stamps) and all(4<=(b-a).total_seconds()<=15 for a,b in zip(stamps,stamps[1:]))
    assert 0<=(now-stamps[-1]).total_seconds()<=120 and 0<=(now-datetime.fromisoformat(review['at'])).total_seconds()<=120,'Stale review'
    assert datetime.fromisoformat(review['at'])>=stamps[-1] and digest(review['native_meta'])==entry['meta_sha256']
    for pid in review['pids_confirmed_absent']:absent(pid)
    used=Path(str(path)+'.used.json');assert not used.exists(),'Review already consumed'
    return review,used


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--deadline',required=True);p.add_argument('--prefix',required=True);p.add_argument('--seconds',type=int,default=60);p.add_argument('--resume',action='store_true');p.add_argument('--prior-state');p.add_argument('--review-stop');a=p.parse_args()
    if a.review_stop and not a.resume:p.error('--review-stop requires --resume')
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo is not None and 1<=a.seconds<=1140
    prefix=Path(a.prefix);prefix.parent.mkdir(parents=True,exist_ok=True);state_path=Path(str(prefix)+'-state.json');assert not any(Path(str(prefix)+tail).exists() for tail in ('-state.json','-guard.json','.log','-resources.jsonl','-audit.json','-native-meta.json')),'Preserve existing evidence; use a unique prefix'
    state=dict(started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),pid=os.getpid(),source_sha256=digest(__file__),status='preflight',goal_status='active',job=a.job)
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save();child=None;handlers={};requested=[]
    try:
        if datetime.now().astimezone()+timedelta(seconds=630)>=deadline:state['status']='refused_deadline';save();return
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:state['status']='refused_busy_global_lock';save();return
            level=pressure();free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]);state.update(initial_pressure=level,initial_free_percent=free)
            if level!=1 or free<55:state['status']='refused_resources';save();return
            assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
            profile_path='profiles/paper_dense_bounded_evaluation.json';profile=load(profile_path);job=next(j for j in profile['jobs'] if j['name']==a.job);assert job['suite'] in ('core','extra','mmmlu'),'Text only; code needs its own generation/judging harness'
            binding=load('runs/paper-dense-evaluation-binding.json');checkpoint=profile['checkpoint'];assert digest('profiles/paper_dense_evaluation.json')==profile['parent_profile_sha256']==binding['profile_sha256']
            assert digest('evaluate_dense_bounded.py')==profile['engine_sha256'] and digest('bounded_dense_inference.py')==profile['kernel_sha256'] and digest(checkpoint)==binding['checkpoint_sha256']
            assert digest(binding['checkpoint_audit'])==binding['checkpoint_audit_sha256'];training=load(binding['checkpoint_audit']);assert training['phase_complete'] and training['step']==8000
            assert all(digest(path)==sha for path,sha in training['source_hashes'].items()) and digest(job['primary_meta'])==job['primary_meta_sha256']
            output=job['output'];exists=Path(output).exists()
            if exists!=a.resume:state['status']='refused_existing_output_or_missing_resume';save();return
            old_bytes=0;old_sha=None;old_items=0;review=None;used=None
            if exists:
                assert a.prior_state,'Resume requires the latest independently audited state'
                prior=load(a.prior_state);absent(prior['pid']);prior_deadline=datetime.fromisoformat(prior['deadline']);assert deadline<=prior_deadline or datetime.now().astimezone()>=prior_deadline,'Do not reset a prior round budget before it expires'
                entry=next(j for j in reversed(prior['jobs']) if j['output']==output);guard=load(entry['guard']);absent(guard['child_pid'])
                assert digest(output)==entry['predictions_sha256'] and digest(output+'.meta.json')==entry['meta_sha256'] and digest(entry['audit_file'])==entry['audit_sha256']
                audit=load(entry['audit_file'])['files'][output];assert audit['file_sha256']==entry['predictions_sha256'] and audit['items']==entry['items']
                meta=load(output+'.meta.json');assert not meta['full_dataset_generated'],'Already complete; do not regenerate'
                signals=guard['signals'];time_only=all(s['signal']=='SIGINT' and s['reason'] in ('wall_budget','deadline') and s.get('pressure')==1 for s in signals)
                genuine_time=guard['returncode']==0 and time_only and (meta['status']=='time_budget' or (meta['status']=='interrupted_signal' and bool(signals)))
                if not genuine_time:
                    resource_stop=meta['status'] in ('memory_pressure','interrupted_memory_pressure') or (meta['status']=='interrupted_signal' and any(s['reason']=='memory_pressure' for s in signals))
                    assert a.review_stop and resource_stop and (guard['returncode']==0 or (guard['returncode'] in (-2,130) and bool(signals))) and all(s['signal']=='SIGINT' and s['reason']=='memory_pressure' and s.get('pressure',0)>=2 for s in signals),'Resource/unknown stop requires a fresh bound resource review'
                    review,used=validate_resource_review(a.review_stop,a.prior_state,entry,profile_path,profile,binding,deadline)
                else:assert not a.review_stop,'Time stop does not consume a resource review'
                assert meta['identity']['checkpoint_sha256']==binding['checkpoint_sha256'] and meta['identity']['content_sha256']==job['primary_content_sha256']
                old_bytes=Path(output).stat().st_size;old_sha=digest(output);old_items=entry['items'];state['prior_state_sha256']=digest(a.prior_state)
            seconds=min(a.seconds,int((deadline-datetime.now().astimezone()).total_seconds())-600)
            if seconds<30:state['status']='refused_deadline';save();return
            command=[sys.executable,'evaluate_dense_bounded.py',checkpoint,'--out',output,'--suite',job['suite'],'--tasks',*job['tasks'],'--full','--tokens',str(job['tokens']),'--context',str(job['context']),'--batch-size',str(job['batch_size']),'--prefill-chunk',str(job['prefill_chunk']),'--gdn-scan',job['gdn_scan'],'--device','cpu','--seconds',str(seconds),'--stop-pressure','2']
            if job['prefix_cache']:command+=['--prefix-cache']
            if job['grouped_gqa']:command+=['--grouped-gqa']
            if exists:command+=['--resume']
            state.update(profile_sha256=digest(profile_path),checkpoint_sha256=binding['checkpoint_sha256'],output=output,preserved_prefix_bytes=old_bytes,preserved_prefix_sha256=old_sha);save()
            if review is not None:
                admission=dict(at=datetime.now().astimezone().isoformat(),prefix=str(prefix),review_file=a.review_stop,review_sha256=digest(a.review_stop),native_status_not_rewritten=True)
                with used.open('x') as stream:stream.write(json.dumps(admission,indent=2)+'\n')
                state['admitted_resource_review']=admission;save()
            guard=dict(started=datetime.now().astimezone().isoformat(),command=command,prefix=str(prefix),signals=[],status='running');guard_path=Path(str(prefix)+'-guard.json')
            for signum in (signal.SIGINT,signal.SIGTERM):handlers[signum]=signal.signal(signum,lambda signum,frame:requested.append(signal.Signals(signum).name))
            with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as resources:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);guard['child_pid']=child.pid;guard_path.write_text(json.dumps(guard,indent=2)+'\n');state.update(status='running',child_pid=child.pid);save();tick=time.monotonic();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure();probe=subprocess.run(['ps','-p',str(child.pid),'-o','pid=,ppid=,rss=,command='],capture_output=True,text=True).stdout
                    resources.write(json.dumps(dict(at=now.isoformat(),pressure=level,child=probe,vm_stat=native(['vm_stat']),swap=native(['sysctl','vm.swapusage'])))+'\n');resources.flush()
                    reason='memory_pressure' if level>=2 else 'parent_signal' if requested else 'deadline' if now>=deadline-timedelta(seconds=600) else 'wall_budget' if time.monotonic()-tick>=1180 else None
                    if stop is None and reason:
                        child.send_signal(signal.SIGINT);stop=time.monotonic();guard['signals'].append(dict(at=now.isoformat(),reason=reason,signal='SIGINT',pressure=level))
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):
                        child.kill();guard['signals'].append(dict(at=now.isoformat(),reason='own_child_grace_exceeded',signal='SIGKILL'));break
                    time.sleep(2)
                child.wait();guard.update(status='terminal',returncode=child.returncode,ended=datetime.now().astimezone().isoformat());guard_path.write_text(json.dumps(guard,indent=2)+'\n');absent(child.pid)
            meta=load(output+'.meta.json');assert meta['identity']['checkpoint_sha256']==binding['checkpoint_sha256'] and meta['identity']['content_sha256']==job['primary_content_sha256']
            assert sum(v['total_dataset'] for v in meta['tasks'].values())==job['target_items']
            with Path(output).open('rb') as stream:
                h=hashlib.sha256();left=old_bytes
                while left:
                    chunk=stream.read(min(left,1048576));assert chunk;h.update(chunk);left-=len(chunk)
                assert not old_bytes or h.hexdigest()==old_sha
            native_meta=Path(str(prefix)+'-native-meta.json');native_meta.write_bytes(Path(output+'.meta.json').read_bytes());entry=dict(name=job['name'],output=output,items=meta['completed_items'],target=job['target_items'],status=meta['status'],full_generated=meta['full_dataset_generated'],guard=str(guard_path),meta_sha256=digest(native_meta),predictions_sha256=digest(output),native_meta=str(native_meta))
            state.update(status='auditing_saved_output',jobs=[entry]);save()
            with Path(str(prefix)+'-audit.log').open('x') as log:subprocess.run([sys.executable,'audit_core_results.py',output,'--suite',job['suite'],'--allow-partial','--out',str(prefix)+'-audit.json'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,int((deadline-datetime.now().astimezone()).total_seconds())-30))
            entry.update(audit_file=str(prefix)+'-audit.json',audit_sha256=digest(str(prefix)+'-audit.json'))
            signals=guard['signals'];time_only=all(s['signal']=='SIGINT' and s['reason'] in ('wall_budget','deadline') and s.get('pressure')==1 for s in signals);safe_native=meta['status'] in ('complete','time_budget') or (meta['status']=='interrupted_signal' and bool(signals) and time_only)
            state.update(status='saved_and_audited' if child.returncode==0 and time_only and safe_native and meta['completed_items']>old_items else 'stop_requires_review',ended=datetime.now().astimezone().isoformat(),goal_status='active');save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='needs_review',error_type=type(error).__name__,error=str(error),ended=datetime.now().astimezone().isoformat());save();raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)
    print(state['status'])


if __name__=='__main__':main()
