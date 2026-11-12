"""One locally guarded formal quality call; full targets and native stops retained."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from qwen35_quality_data import PROTOCOL, load, digest, selection, training_binding, write_json
from run_dense_text_guarded import pressure, native, absent


def prior_identity(path, profile, audit, checkpoint, arm, job, deadline):
    prior = load(path)
    for pid in [prior['pid'], prior['child_pid']]:absent(pid)
    assert deadline <= datetime.fromisoformat(prior['deadline']) or datetime.now().astimezone() >= datetime.fromisoformat(prior['deadline'])
    binding = training_binding(profile, audit, checkpoint, arm)
    assert prior['binding'] == binding and prior['job'] == job
    output = selection(load(PROTOCOL), arm, job)['output_template'].format(arm=arm)
    assert prior['output'] == output
    for file, sha in [(output, prior['predictions_sha256']), (output+'.meta.json', prior['meta_sha256']), (prior['audit_file'], prior['audit_sha256'])]:
        assert digest(file) == sha
    receipt = load(prior['audit_file'])
    assert receipt['status'] == 'passed_source_output_audit' and not receipt['full_dataset_generated']
    assert receipt['identity']['binding'] == binding
    return prior, dict(prior_state_sha256=digest(path), binding=binding, job=job,
                       predictions_sha256=prior['predictions_sha256'],meta_sha256=prior['meta_sha256'],
                       audit_sha256=prior['audit_sha256'],guard_sha256=digest(__file__),
                       deadline=deadline.isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['arm','job','training-profile','checkpoint','audit','deadline','prefix']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--prior-state')
    parser.add_argument('--prepare-review',action='store_true');parser.add_argument('--review-stop')
    args=parser.parse_args();deadline=datetime.fromisoformat(args.deadline)
    assert deadline.tzinfo and 30 <= args.seconds <= 1140
    assert args.resume == bool(args.prior_state)
    assert not args.prepare_review or (args.resume and not args.review_stop)
    prefix=Path(args.prefix);assert prefix.parent.resolve()==Path('runs').resolve()
    assert not any(Path(str(prefix)+tail).exists() for tail in ['-state.json','.log','-audit.json','-resources.jsonl'])
    state=dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),
               goal_status='active',status='preflight',signals=[],job=args.job)
    def save():write_json(str(prefix)+'-state.json',state)
    save();child=None;handlers={};requests=[];tick=time.monotonic()
    try:
        assert datetime.now().astimezone()+timedelta(seconds=660)<deadline<=datetime.now().astimezone()+timedelta(hours=2)
        with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:state['status']='refused_busy_global_lock';save();return
            assert all(os.environ.get(key) is None for key in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
            binding=training_binding(args.training_profile,args.audit,args.checkpoint,args.arm)
            job=selection(load(PROTOCOL),args.arm,args.job);output=job['output_template'].format(arm=args.arm)
            state.update(binding=binding,output=output)
            review_used=None
            if args.resume:
                prior,identity=prior_identity(args.prior_state,args.training_profile,args.audit,args.checkpoint,args.arm,args.job,deadline)
                if args.prepare_review:
                    assert prior['status']=='stop_requires_review' and prior['native_status'] in ['memory_pressure','interrupted_signal']
                    assert prior['native_status']=='memory_pressure' or any(s['reason']=='memory_pressure' for s in prior['signals'])
                    assert all(s['reason']=='memory_pressure' and s['signal']=='SIGINT' for s in prior['signals'])
                    history=[];good=[];start=time.monotonic()
                    while len(good)<12 and time.monotonic()-start<300:
                        assert datetime.now().astimezone()+timedelta(seconds=660)<deadline
                        row=dict(at=datetime.now().astimezone().isoformat(),pressure=pressure(),free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]),swap=native(['sysctl','vm.swapusage']))
                        history.append(row);good=good+[row] if row['pressure']==1 and row['free']>=55 else []
                        write_json(str(prefix)+'-observations.json',dict(all_samples=history,consecutive_good=len(good)))
                        if len(good)<12:time.sleep(5)
                    assert len(good)==12
                    assert prior_identity(args.prior_state,args.training_profile,args.audit,args.checkpoint,args.arm,args.job,deadline)[1]==identity
                    review=str(prefix)+'-review.json';write_json(review,dict(identity=identity,samples=good,observations_sha256=digest(str(prefix)+'-observations.json'),observations_file=str(prefix)+'-observations.json'))
                    state.update(status='resource_review_created_no_model_child',review_file=review);save();return
                known_time = (prior['status']=='stop_requires_review' and prior['returncode']==0
                              and prior['native_status']=='interrupted_signal' and prior['signals']
                              and all(s['signal']=='SIGINT' and s['reason'] in ['deadline','wall_budget'] and s.get('pressure')==1 for s in prior['signals']))
                if prior['status']=='saved_and_audited' or known_time:assert not args.review_stop
                else:
                    assert args.review_stop and prior['status']=='stop_requires_review'
                    review=load(args.review_stop);assert review['identity']==identity
                    samples=review['samples'];assert len(samples)==12 and all(row['pressure']==1 and row['free']>=55 for row in samples)
                    stamps=[datetime.fromisoformat(row['at']) for row in samples]
                    assert all(4<=(b-a).total_seconds()<=15 for a,b in zip(stamps,stamps[1:]))
                    assert 0<=(datetime.now().astimezone()-stamps[-1]).total_seconds()<=120
                    history=load(review['observations_file']);assert digest(review['observations_file'])==review['observations_sha256'] and history['all_samples'][-12:]==samples and history['consecutive_good']==12
                    review_used=Path(args.review_stop+'.used.json');assert not review_used.exists()
            else:assert not Path(output).exists() and not Path(output+'.meta.json').exists() and not args.review_stop
            level=pressure();free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1])
            state.update(initial_pressure=level,initial_free_percent=free)
            if level!=1 or free<55:state['status']='refused_resources';save();return
            seconds=min(args.seconds,1140-int(time.monotonic()-tick),int((deadline-datetime.now().astimezone()).total_seconds())-600);assert seconds>=30
            command=[sys.executable,'evaluate_qwen35_quality.py','--arm',args.arm,'--job',args.job,'--training-profile',args.training_profile,'--checkpoint',args.checkpoint,'--audit',args.audit,'--seconds',str(seconds),'--deadline',(deadline-timedelta(seconds=600)).isoformat()]+(['--resume'] if args.resume else [])
            for signum in [signal.SIGINT,signal.SIGTERM]:handlers[signum]=signal.signal(signum,lambda n,f:requests.append(n))
            with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as resources:
                if review_used is not None:
                    with review_used.open('x') as stream:json.dump(dict(at=datetime.now().astimezone().isoformat(),review_sha256=digest(args.review_stop),native_status_not_rewritten=True),stream)
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'QWEN35_QUALITY_GUARD_PID':str(os.getpid())})
                state.update(status='running',child_pid=child.pid,command=command);save();stop=None
                while child.poll() is None:
                    now=datetime.now().astimezone();level=pressure()
                    resources.write(json.dumps(dict(at=now.isoformat(),pressure=level,swap=native(['sysctl','vm.swapusage'])))+'\n');resources.flush()
                    reason='memory_pressure' if level>=2 else 'parent_signal' if requests else 'wall_budget' if time.monotonic()-tick>=1180 else 'deadline' if now>=deadline-timedelta(seconds=600) else None
                    if stop is None and reason:
                        child.send_signal(signal.SIGINT);stop=time.monotonic();state['signals'].append(dict(at=now.isoformat(),reason=reason,signal='SIGINT',pressure=level));save()
                    elif stop is not None and (time.monotonic()-stop>=60 or time.monotonic()-tick>=1198):
                        child.kill();state['signals'].append(dict(at=now.isoformat(),reason='own_child_grace_exceeded',signal='SIGKILL'));break
                    time.sleep(2)
                child.wait();absent(child.pid);state.update(returncode=child.returncode,status='terminal');save()
            assert child.returncode==0,'Worker failed; preserve native output and inspect before resuming'
            meta=load(output+'.meta.json');assert meta['status'] in ['complete','time_budget','memory_pressure','interrupted_signal']
            receipt=str(prefix)+'-audit.json'
            with Path(str(prefix)+'-audit.log').open('x') as log:
                subprocess.run([sys.executable,'audit_qwen35_quality.py',output,'--allow-partial','--out',receipt],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            audited=load(receipt);assert audited['items']> (load(args.prior_state)['items'] if args.resume else 0),'No progress; diagnose before resuming'
            write_json(str(prefix)+'-native-meta.json',meta)
            safe=not state['signals'] and meta['status'] in ['complete','time_budget']
            state.update(status='saved_and_audited' if safe else 'stop_requires_review',native_status=meta['status'],items=audited['items'],full_dataset_generated=audited['full_dataset_generated'],predictions_sha256=digest(output),meta_sha256=digest(output+'.meta.json'),audit_file=receipt,audit_sha256=digest(receipt));save()
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
