"""Create a fresh one-use Dense text resource review; never starts a model."""
import argparse,fcntl,json,re,time
from pathlib import Path
from datetime import datetime,timedelta
from run_dense_text_guarded import load,digest,absent,native,pressure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior-state',required=True);p.add_argument('--job',required=True);p.add_argument('--deadline',required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
    now=datetime.now().astimezone();deadline=datetime.fromisoformat(a.deadline)
    assert deadline.tzinfo and now+timedelta(seconds=660)<deadline<=now+timedelta(hours=2),'Admission deadline leaves no model/cleanup budget'
    paths={key:Path(a.prefix+tail) for key,tail in [('observations','-observations.json'),('samples','-samples.json'),('review','-review.json')]}
    assert all(not path.exists() for path in paths.values()),'Use a unique prefix'
    prior=load(a.prior_state);assert prior['status']=='stop_requires_review'
    assert deadline<=datetime.fromisoformat(prior['deadline']) or now>=datetime.fromisoformat(prior['deadline']),'Do not reset a live round deadline'
    entry=next(j for j in reversed(prior['jobs']) if j['name']==a.job);guard=load(entry['guard'])
    assert entry['status'] in ('memory_pressure','interrupted_memory_pressure') or (entry['status']=='interrupted_signal' and any(s['reason']=='memory_pressure' for s in guard['signals']))
    assert guard['returncode'] in (0,-2,130) and all(s['signal']=='SIGINT' and s['reason']=='memory_pressure' and s.get('pressure',0)>=2 for s in guard['signals'])
    if guard['returncode']!=0:assert guard['signals']
    profile_path='profiles/paper_dense_bounded_evaluation.json';profile=load(profile_path);binding=load('runs/paper-dense-evaluation-binding.json')
    def verify():
        for path,sha in [(entry['output'],entry['predictions_sha256']),(entry['output']+'.meta.json',entry['meta_sha256']),(entry['native_meta'],entry['meta_sha256']),(entry['audit_file'],entry['audit_sha256'])]:assert digest(path)==sha
        assert digest('evaluate_dense_bounded.py')==profile['engine_sha256'] and digest('bounded_dense_inference.py')==profile['kernel_sha256'] and digest(binding['checkpoint'])==binding['checkpoint_sha256']
        assert load(entry['audit_file'])['files'][entry['output']]['items']==entry['items']
    verify();pids=list({prior['pid'],guard['child_pid']})
    for pid in pids:absent(pid)
    identity=dict(state_file=a.prior_state,state_sha256=digest(a.prior_state),output=entry['output'],predictions_sha256=entry['predictions_sha256'],meta_sha256=entry['meta_sha256'],audit_file=entry['audit_file'],audit_sha256=entry['audit_sha256'],profile_sha256=digest(profile_path),engine_sha256=profile['engine_sha256'],kernel_sha256=profile['kernel_sha256'],checkpoint_sha256=binding['checkpoint_sha256'],guard_source_sha256=digest('run_dense_text_guarded.py'),deadline=deadline.isoformat(),native_meta=entry['native_meta'],pids_confirmed_absent=pids)
    history=[];samples=[];tick=time.monotonic()
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while len(samples)<12 and time.monotonic()-tick<300:
            assert datetime.now().astimezone()+timedelta(seconds=660)<deadline
            row=dict(at=datetime.now().astimezone().isoformat(),pressure=pressure(),free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]),swap=native(['sysctl','vm.swapusage']))
            history.append(row);samples=samples+[row] if row['pressure']==1 and row['free']>=55 else []
            paths['observations'].write_text(json.dumps(dict(all_samples=history,consecutive_good=len(samples)),indent=2)+'\n')
            if len(samples)<12:time.sleep(5)
        assert len(samples)==12,'No stable recovery window; no review issued'
        verify();assert digest(a.prior_state)==identity['state_sha256'] and digest(profile_path)==identity['profile_sha256'] and digest('run_dense_text_guarded.py')==identity['guard_source_sha256']
        for pid in pids:absent(pid)
        paths['samples'].write_text(json.dumps(samples,indent=2)+'\n')
        review=dict(at=datetime.now().astimezone().isoformat(),**identity,samples_file=str(paths['samples']),samples_sha256=digest(paths['samples']),observation_file=str(paths['observations']),observation_sha256=digest(paths['observations']),builder_sha256=digest(__file__),native_status_not_rewritten=True)
        with paths['review'].open('x') as f:json.dump(review,f,indent=2)
    print(paths['review'])


if __name__=='__main__':main()
