"""One-use resource admission for both fixed formal phases; no model imports."""
import json,re,time
from pathlib import Path
from datetime import datetime,timedelta
from qwen35_formal_contract import contract
from run_dense_text_guarded import load,digest,absent,native,pressure


def identity(profile_path,prior_path,deadline):
    p=load(profile_path);contract(p,p['stage']);prior=load(prior_path);audit=load(prior['audit_file'])
    assert deadline<=datetime.fromisoformat(prior['deadline']) or datetime.now().astimezone()>=datetime.fromisoformat(prior['deadline'])
    assert prior['status']=='stop_requires_review' and prior['returncode'] in [0,-2,130] and prior['signals']
    assert all(s['signal']=='SIGINT' and s['reason']=='memory_pressure' and s.get('pressure',0)>=2 for s in prior['signals'])
    assert audit['status']=='passed_checkpoint_checks' and not audit['phase_complete'] and audit['step']==prior['step']<p['steps'] and audit['target_steps']==p['steps']
    checkpoint=Path(p['out'])/'last.pt';assert checkpoint.resolve()==Path(prior['checkpoint']).resolve()
    assert digest(checkpoint)==prior['checkpoint_sha256']==audit['checkpoint_sha256']
    assert digest(profile_path)==prior['profile_sha256']==audit['profile_sha256'] and digest(prior['audit_file'])==prior['audit_sha256']
    for pid in [prior['pid'],prior['child_pid']]:absent(pid)
    return dict(prior_state=str(Path(prior_path).resolve()),prior_state_sha256=digest(prior_path),profile_sha256=digest(profile_path),checkpoint_sha256=prior['checkpoint_sha256'],audit_sha256=prior['audit_sha256'],source_hashes=p['source_hashes'],guard_sha256=digest('run_qwen35_formal_guarded.py'),protocol_sha256=digest(__file__),stage=p['stage'],input_step=prior['step'],deadline=deadline.isoformat())


def prepare(profile_path,prior_path,deadline,prefix):
    # The caller holds the same global model lock for this entire observation.
    expected=identity(profile_path,prior_path,deadline);paths={k:Path(str(prefix)+suffix) for k,suffix in [('history','-observations.json'),('samples','-samples.json'),('review','-review.json')]};assert not any(p.exists() for p in paths.values())
    history=[];samples=[];tick=time.monotonic()
    while len(samples)<12 and time.monotonic()-tick<300:
        assert datetime.now().astimezone()+timedelta(seconds=660)<deadline
        row=dict(at=datetime.now().astimezone().isoformat(),pressure=pressure(),free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]),swap=native(['sysctl','vm.swapusage']))
        history.append(row);samples=samples+[row] if row['pressure']==1 and row['free']>=55 else []
        paths['history'].write_text(json.dumps(dict(all_samples=history,consecutive_good=len(samples)),indent=2)+'\n')
        if len(samples)<12:time.sleep(5)
    assert len(samples)==12 and identity(profile_path,prior_path,deadline)==expected
    paths['samples'].write_text(json.dumps(samples,indent=2)+'\n')
    receipt=dict(at=datetime.now().astimezone().isoformat(),**expected,samples_file=str(paths['samples']),samples_sha256=digest(paths['samples']),observations_file=str(paths['history']),observations_sha256=digest(paths['history']),native_status_not_rewritten=True)
    with paths['review'].open('x') as f:json.dump(receipt,f,indent=2)
    return str(paths['review'])


def admit(path,profile_path,prior_path,deadline):
    expected=identity(profile_path,prior_path,deadline);r=load(path);now=datetime.now().astimezone();assert all(r[k]==v for k,v in expected.items())
    assert digest(r['samples_file'])==r['samples_sha256'] and digest(r['observations_file'])==r['observations_sha256']
    samples=load(r['samples_file']);history=load(r['observations_file']);assert len(samples)==12 and history['all_samples'][-12:]==samples and history['consecutive_good']==12
    assert all(s['pressure']==1 and s['free']>=55 for s in samples)
    stamps=[datetime.fromisoformat(s['at']) for s in samples];assert all(t.tzinfo for t in stamps) and all(4<=(b-a).total_seconds()<=15 for a,b in zip(stamps,stamps[1:]))
    assert 0<=(now-stamps[-1]).total_seconds()<=120 and 0<=(now-datetime.fromisoformat(r['at'])).total_seconds()<=120 and datetime.fromisoformat(r['at'])>=stamps[-1]
    used=Path(str(path)+'.used.json');assert not used.exists(),'Review already consumed'
    return r,used
