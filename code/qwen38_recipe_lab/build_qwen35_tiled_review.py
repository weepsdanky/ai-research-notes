"""Fresh one-use resource review for the unchanged tiled numerical worker."""
import argparse,fcntl,json,re,time
from pathlib import Path
from datetime import datetime,timedelta
from run_qwen35_long_reviewed import review_identity as frozen_identity,admit_review as frozen_admit
from run_dense_text_guarded import load,digest,pressure,native


def review_identity(profile_path,prior_path,deadline):
    identity=frozen_identity(profile_path,prior_path,deadline);profile=load(profile_path)
    assert profile['training_execution']=='tiled_streamed_one_update_separate_validation' and profile['runtime_tiling']==dict(query_chunk=32,head_chunk=8)
    assert Path(load(prior_path)['checkpoint']).resolve()==(Path(profile['out'])/'last.pt').resolve()
    identity['runtime_review_sources']={p:digest(p) for p in ['build_qwen35_tiled_review.py','run_qwen35_tiled_reviewed.py','run_qwen35_long_reviewed.py']}
    return identity


def admit_review(path,profile_path,prior_path,deadline):
    expected=review_identity(profile_path,prior_path,deadline);review,used=frozen_admit(path,profile_path,prior_path,deadline)
    assert all(review[k]==v for k,v in expected.items()),'Tiled runtime review identity mismatch'
    return review,used


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--prior-state',required=True);p.add_argument('--deadline',required=True);p.add_argument('--prefix',required=True);a=p.parse_args();deadline=datetime.fromisoformat(a.deadline)
    assert deadline.tzinfo and datetime.now().astimezone()+timedelta(seconds=660)<deadline<=datetime.now().astimezone()+timedelta(hours=2)
    prior=load(a.prior_state);assert deadline<=datetime.fromisoformat(prior['deadline']) or datetime.now().astimezone()>=datetime.fromisoformat(prior['deadline'])
    identity=review_identity(a.profile,a.prior_state,deadline);history_path=Path(a.prefix+'-observations.json');samples_path=Path(a.prefix+'-samples.json');review_path=Path(a.prefix+'-review.json');assert not any(p.exists() for p in [history_path,samples_path,review_path])
    history=[];samples=[];tick=time.monotonic()
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while len(samples)<12 and time.monotonic()-tick<300:
            assert datetime.now().astimezone()+timedelta(seconds=660)<deadline
            row=dict(at=datetime.now().astimezone().isoformat(),pressure=pressure(),free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1]),swap=native(['sysctl','vm.swapusage']))
            history.append(row);samples=samples+[row] if row['pressure']==1 and row['free']>=55 else []
            history_path.write_text(json.dumps(dict(all_samples=history,consecutive_good=len(samples)),indent=2)+'\n')
            if len(samples)<12:time.sleep(5)
        assert len(samples)==12,'No stable resource window'
        assert review_identity(a.profile,a.prior_state,deadline)==identity
        samples_path.write_text(json.dumps(samples,indent=2)+'\n')
        receipt=dict(at=datetime.now().astimezone().isoformat(),**identity,samples_file=str(samples_path),samples_sha256=digest(samples_path),observations_file=str(history_path),observations_sha256=digest(history_path),native_status_not_rewritten=True,numerical_engine_unchanged=True)
        with review_path.open('x') as stream:json.dump(receipt,stream,indent=2)
    print(review_path)


if __name__=='__main__':main()
