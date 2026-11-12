"""Guard the separate32K phase without changing completed4K source bindings."""
import argparse,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta
from run_dense_text_guarded import digest,load,native,pressure,absent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--deadline',required=True);p.add_argument('--prefix',required=True);p.add_argument('--seconds',type=int,default=60);p.add_argument('--resume',action='store_true');p.add_argument('--prior-state');a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo and 30<=a.seconds<=1140
    prefix=Path(a.prefix);assert prefix.parent.resolve()==Path('runs').resolve()
    assert not any(Path(str(prefix)+s).exists() for s in ['-state.json','.log','-resources.jsonl','-audit.json'])
    state_path=Path(str(prefix)+'-state.json');state=dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),status='preflight',signals=[],source_sha256=digest(__file__),goal_status='active')
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save();child=None;handlers={};requests=[];tick=time.monotonic()
    try:
        assert datetime.now().astimezone()+timedelta(seconds=660)<deadline<=datetime.now().astimezone()+timedelta(hours=2)
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:state['status']='refused_busy_global_lock';save();return
            free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]);level=pressure()
            state.update(initial_pressure=level,initial_free_percent=free)
            if level!=1 or free<55:state['status']='refused_resources';save();return
            assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
            profile=load(a.profile);assert profile['status']=='resource_probe_candidate' and profile['scope']=='Mac public-family control, not private Table1'
            assert all(digest(p)==sha for p,sha in profile['source_hashes'].items())
            assert all(digest(p)==sha for p,sha in profile['component_receipts'].items())
            assert all(digest(row['directory']+'/meta.json')==row['meta_sha256'] for row in profile['data'])
            checkpoint=Path(profile['out'])/'last.pt';state.update(profile_sha256=digest(a.profile),checkpoint=str(checkpoint),initial_pressure=1,initial_free_percent=free,cleanup_reserve_seconds=600)
            if a.resume:
                assert a.prior_state;prior=load(a.prior_state);absent(prior['pid']);absent(prior['child_pid'])
                assert prior['status']=='saved_and_audited' and prior['returncode']==0 and not prior['signals'],'Resource/unknown stop requires fresh review; this probe entry has no resource override'
                assert deadline<=datetime.fromisoformat(prior['deadline']) or datetime.now().astimezone()>=datetime.fromisoformat(prior['deadline'])
                assert prior['profile_sha256']==digest(a.profile) and digest(checkpoint)==prior['checkpoint_sha256']
                assert digest(prior['audit_file'])==prior['audit_sha256']
                audit=load(prior['audit_file']);assert audit['status']=='passed_checkpoint_checks' and not audit['phase_complete']
                assert audit['checkpoint_sha256']==digest(checkpoint) and audit['profile_sha256']==digest(a.profile)
                state['prior_state_sha256']=digest(a.prior_state)
            else:assert not Path(profile['out']).exists(),'Never overwrite an existing run'
            seconds=min(a.seconds,1140-int(time.monotonic()-tick),int((deadline-datetime.now().astimezone()).total_seconds())-600);assert seconds>=30
            command=[sys.executable,'qwen35_long_layer_checkpoint.py','--profile',a.profile,'--seconds',str(seconds)]+(['--resume'] if a.resume else [])
            for signum in [signal.SIGINT,signal.SIGTERM]:handlers[signum]=signal.signal(signum,lambda n,f:requests.append(n))
            with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as resources:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'QWEN35_CONTROL_GUARD_PID':str(os.getpid())});state.update(status='running',child_pid=child.pid,command=command);save();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure();probe=subprocess.run(['ps','-p',str(child.pid),'-o','pid=,ppid=,rss=,command='],capture_output=True,text=True).stdout
                    resources.write(json.dumps(dict(at=now.isoformat(),pressure=level,child=probe,swap=native(['sysctl','vm.swapusage'])))+'\n');resources.flush()
                    reason='memory_pressure' if level>=2 else 'parent_signal' if requests else 'wall_budget' if time.monotonic()-tick>=1180 else 'deadline' if now>=deadline-timedelta(seconds=600) else None
                    if stop is None and reason:
                        child.send_signal(signal.SIGINT);stop=time.monotonic();state['signals'].append(dict(at=now.isoformat(),reason=reason,signal='SIGINT',pressure=level));save()
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):
                        child.kill();state['signals'].append(dict(at=now.isoformat(),reason='own_child_grace_exceeded',signal='SIGKILL'));break
                    time.sleep(2)
                child.wait();absent(child.pid);state.update(status='terminal',returncode=child.returncode,ended=datetime.now().astimezone().isoformat());save()
            if checkpoint.exists():
                audit_file=str(prefix)+'-audit.json'
                with Path(str(prefix)+'-audit.log').open('x') as log:subprocess.run([sys.executable,'audit_qwen35_long_phase.py','--profile',a.profile,'--checkpoint',str(checkpoint),'--out',audit_file],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
                audit=load(audit_file);state.update(audit_file=audit_file,audit_sha256=digest(audit_file),checkpoint_sha256=digest(checkpoint),step=audit['step'],target_steps=audit['target_steps'],phase_complete=audit['phase_complete'])
                state['status']='saved_and_audited' if child.returncode==0 and not state['signals'] else 'stop_requires_review';save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='needs_review',error_type=type(error).__name__,error=str(error));save();raise
    finally:
        for n,h in handlers.items():signal.signal(n,h)


if __name__=='__main__':main()
