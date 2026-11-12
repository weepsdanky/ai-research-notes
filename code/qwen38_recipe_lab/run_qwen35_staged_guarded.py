"""Native guard for the separately profiled staged formal continuation."""
import argparse,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta
from qwen35_quality_data import load,digest,write_json
from run_dense_text_guarded import native,pressure,absent
from qwen35_formal_contract import contract
from qwen35_formal_review import prepare,admit


def pending_identity(out,prior):
    path=out/'pending.pt';sha=digest(path) if path.exists() else None
    assert sha==prior.get('pending_sha256')
    if sha is not None:
        receipt=load(prior['pending_audit']);assert digest(prior['pending_audit'])==prior['pending_audit_sha256']
        assert receipt['status']=='passed_pending_state_checks' and receipt['pending_sha256']==sha and receipt['checkpoint_sha256']==prior['checkpoint_sha256']
    return dict(pending_sha256=sha,pending_audit_sha256=prior.get('pending_audit_sha256'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['profile','deadline','prefix']:p.add_argument('--'+k,required=True)
    p.add_argument('--seconds',type=int,default=60);p.add_argument('--prior-state');p.add_argument('--review-stop');p.add_argument('--prepare-review',action='store_true');p.add_argument('--migration-only',action='store_true');a=p.parse_args()
    assert 30<=a.seconds<=1140 and (a.migration_only!=bool(a.prior_state))
    assert not a.prepare_review or (a.prior_state and not a.review_stop)
    deadline=datetime.fromisoformat(a.deadline);prefix=Path(a.prefix);assert prefix.parent.resolve()==Path('runs').resolve()
    assert not any(Path(str(prefix)+tail).exists() for tail in ['-state.json','.log','-resources.jsonl','-audit.json','-pending-audit.json'])
    state=dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),status='preflight',signals=[],goal_status='active',source_sha256=digest(__file__))
    def save():write_json(str(prefix)+'-state.json',state)
    save();child=None;handlers={};requests=[];tick=time.monotonic()
    try:
        assert datetime.now().astimezone()+timedelta(seconds=660)<deadline<=datetime.now().astimezone()+timedelta(hours=2)
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            level=pressure();free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]);state.update(initial_pressure=level,initial_free_percent=free)
            if not a.prepare_review and (level!=1 or free<55):state['status']='refused_resources';save();return
            assert all(os.environ.get(k) is None for k in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
            profile=load(a.profile);contract(profile,'formal_long');assert profile['training_execution']=='formal_staged_update'
            assert {'run_qwen35_staged_guarded.py','qwen35_staged_formal_long.py','staged_qwen35_update.py','audit_qwen35_staged_pending.py','audit_qwen35_formal_long.py'}<=profile['source_hashes'].keys()
            out=Path(profile['out']);cp=out/'last.pt';state.update(stage='formal_long',checkpoint=str(cp),profile_sha256=digest(a.profile),cleanup_reserve_seconds=600)
            used=None;prior=None;old_units=0
            if a.migration_only:
                m=profile['migration'];assert not out.exists() and a.review_stop and digest(m['legacy_state'])==m['legacy_state_sha256']
                review,used=admit(a.review_stop,m['legacy_profile'],m['legacy_state'],deadline);input_step=m['credit_steps']
            else:
                prior=load(a.prior_state);absent(prior['pid']);absent(prior['child_pid']);assert prior['profile_sha256']==state['profile_sha256'] and digest(cp)==prior['checkpoint_sha256']
                assert deadline<=datetime.fromisoformat(prior['deadline']) or datetime.now().astimezone()>=datetime.fromisoformat(prior['deadline'])
                assert digest(prior['audit_file'])==prior['audit_sha256'];accepted=load(prior['audit_file'])
                assert accepted['status']=='passed_checkpoint_checks' and accepted['checkpoint_sha256']==prior['checkpoint_sha256'] and accepted['step']==prior['step']<20
                pending=pending_identity(out,prior);old_units=prior.get('pending_units',0);input_step=prior['step']
                if a.prepare_review:
                    base=prepare(a.profile,a.prior_state,deadline,prefix);assert pending_identity(out,prior)==pending
                    review_path=str(prefix)+'-staged-review.json';assert not Path(review_path).exists()
                    write_json(review_path,dict(base_review=base,base_review_sha256=digest(base),prior_state_sha256=digest(a.prior_state),staged_guard_sha256=digest(__file__),**pending))
                    state.update(status='resource_review_created_no_model_child',review_file=review_path);save();return
                ordinary=prior['status'] in ['migration_checked_no_updates','saved_and_audited','staged_partial_audited'] and prior['returncode']==0 and not prior['signals']
                known_time=prior['status']=='stop_requires_review' and prior['returncode'] in [0,-2,130] and prior['signals'] and all(s['signal']=='SIGINT' and s['reason'] in ['deadline','wall_budget'] and s.get('pressure')==1 for s in prior['signals'])
                if ordinary or known_time:assert not a.review_stop
                else:
                    assert a.review_stop
                    ext=load(a.review_stop);assert ext['prior_state_sha256']==digest(a.prior_state) and ext['staged_guard_sha256']==digest(__file__) and all(ext[k]==v for k,v in pending.items())
                    assert digest(ext['base_review'])==ext['base_review_sha256'];review,used=admit(ext['base_review'],a.profile,a.prior_state,deadline)
                state['prior_state_sha256']=digest(a.prior_state)
            state['input_step']=input_step
            seconds=min(a.seconds,1140-int(time.monotonic()-tick),int((deadline-datetime.now().astimezone()).total_seconds())-600);assert seconds>=30
            cmd=[sys.executable,'qwen35_staged_formal_long.py','--profile',a.profile,'--seconds',str(seconds)]+(['--migration-only'] if a.migration_only else [])
            for signum in [signal.SIGINT,signal.SIGTERM]:handlers[signum]=signal.signal(signum,lambda n,f:requests.append(n))
            with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as resources:
                if used is not None:
                    admission=dict(at=datetime.now().astimezone().isoformat(),review_file=a.review_stop,review_sha256=digest(a.review_stop),native_status_not_rewritten=True)
                    with used.open('x') as f:json.dump(admission,f,indent=2)
                    state['admitted_resource_review']=admission;save()
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'QWEN35_STAGED_FORMAL_GUARD_PID':str(os.getpid())});state.update(status='running',child_pid=child.pid,command=cmd);save();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure();rss=subprocess.run(['ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True).stdout.strip()
                    resources.write(json.dumps(dict(at=now.isoformat(),pressure=level,rss_kib=rss,swap=native(['sysctl','vm.swapusage'])))+'\n');resources.flush()
                    reason='memory_pressure' if level>=2 else 'parent_signal' if requests else 'wall_budget' if time.monotonic()-tick>=1180 else 'deadline' if now>=deadline-timedelta(seconds=600) else None
                    if stop is None and reason:
                        child.send_signal(signal.SIGINT);stop=time.monotonic();state['signals'].append(dict(at=now.isoformat(),reason=reason,signal='SIGINT',pressure=level));save()
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):child.kill();state['signals'].append(dict(reason='own_child_grace_exceeded',signal='SIGKILL'));break
                    time.sleep(2)
                child.wait();absent(child.pid);state.update(returncode=child.returncode,status='terminal');save()
            assert cp.exists();sha=digest(cp)
            if prior is None or sha!=prior['checkpoint_sha256']:
                audit_file=str(prefix)+'-audit.json'
                with Path(str(prefix)+'-audit.log').open('x') as log:
                    subprocess.run([sys.executable,'audit_qwen35_formal_long.py','--profile',a.profile,'--checkpoint',str(cp),'--out',audit_file],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
            else:audit_file=prior['audit_file']
            audit=load(audit_file);assert audit['status']=='passed_checkpoint_checks' and audit['checkpoint_sha256']==sha
            accepted=audit['step']-input_step;assert 0<=accepted<=1
            state.update(audit_file=audit_file,audit_sha256=digest(audit_file),checkpoint_sha256=sha,step=audit['step'],target_steps=20,phase_complete=audit['phase_complete'],accepted_new_updates=accepted)
            if (out/'pending.pt').exists():
                assert not accepted and not a.migration_only
                pending_file=str(prefix)+'-pending-audit.json'
                with Path(str(prefix)+'-pending-audit.log').open('x') as log:
                    subprocess.run([sys.executable,'audit_qwen35_staged_pending.py','--profile',a.profile,'--out',pending_file],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
                r=load(pending_file);assert r['status']=='passed_pending_state_checks' and r['checkpoint_sha256']==sha and r['units']>=old_units
                state.update(pending_audit=pending_file,pending_audit_sha256=digest(pending_file),pending_sha256=r['pending_sha256'],pending_units=r['units'],pending_phase=r['phase'],pending_ce_chunks=r['ce_chunks'])
            safe=child.returncode==0 and not state['signals']
            if safe and a.migration_only:
                assert not accepted and load(out/'migration.json')['model_optimizer_rng_corpora_bitwise_preserved'];state['status']='migration_checked_no_updates'
            elif safe and accepted:state['status']='saved_and_audited'
            elif safe and state.get('pending_units',0)>old_units:state['status']='staged_partial_audited'
            else:state['status']='stop_requires_review'
            save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='needs_review',error=repr(error));save();raise
    finally:
        state['ended']=datetime.now().astimezone().isoformat();save()
        for number,handler in handlers.items():signal.signal(number,handler)


if __name__=='__main__':main()
