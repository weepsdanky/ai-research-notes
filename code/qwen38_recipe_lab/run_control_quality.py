"""Run one public benchmark across a Mac paper control family."""
import argparse
from datetime import datetime, timedelta
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from audit_control_coverage import SCOPE, arms_for
from control_eval_binding import EXPERIMENTS
from evaluate_ngram_placement import training_input
from train_ngram_capacity import FAMILIES
from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure

RUNS = Path('runs')
LOCK = RUNS / 'hf-mmmlu-global.lock'


def now():
    return datetime.now().astimezone()


def memory():
    free = int(re.search(r'free percentage:\s*(\d+)%', native(['memory_pressure', '-Q']))[1])
    return dict(at=now().isoformat(), level=pressure(), free=free)


def normal(sample):
    return sample['level'] == 1 and sample['free'] >= 55


def review(state, receipt, deadline, paths):
    before = {str(path): digest(path) for path in paths}
    record = dict(paths=before, samples=[], consecutive_good=0)
    state['reviews'].append(record)
    cutoff = min(now()+timedelta(seconds=180), deadline-timedelta(seconds=690))
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while now() < cutoff:
            sample = memory()
            record['samples'].append(sample)
            record['consecutive_good'] = record['consecutive_good']+1 if normal(sample) else 0
            write_json(receipt, state)
            if record['consecutive_good'] == 12:
                assert all(digest(path)==sha for path,sha in before.items())
                return True
            time.sleep(5)
    return False


def execute(args):
    spec = load(SCOPE)['families'][args.experiment]
    suffix = spec.get('aliases', {}).get(args.job, args.job)
    experiment = args.experiment if args.experiment in EXPERIMENTS else None
    capacity = args.experiment if args.experiment in FAMILIES else None
    family_args = []
    if experiment:
        family_args = ['--experiment', experiment]
    if capacity:
        family_args = ['--capacity-family', capacity]
    state = dict(status='running', pid=os.getpid(), started=now().isoformat(),
                 experiment=args.experiment, job=args.job, output_suffix=suffix, deadline=args.deadline.isoformat(),
                 prefix=str(args.prefix), training_audit_sha256=digest(args.training_audit),
                 calls=[], reviews=[], arms={})
    write_json(args.receipt, state)

    def call(tag, command, *, check=True):
        row = dict(tag=tag, args=[str(x) for x in command], started=now().isoformat())
        state['calls'].append(row)
        log = args.receipt.with_name(f'{args.receipt.stem}-{tag}.log')
        with log.open('x') as out:
            child = subprocess.Popen([sys.executable,*row['args']], stdout=out, stderr=subprocess.STDOUT)
            row.update(pid=child.pid, log=str(log))
            write_json(args.receipt, state)
            row['returncode'] = child.wait()
        row['ended'] = now().isoformat()
        write_json(args.receipt, state)
        if check:
            assert row['returncode']==0, log
        return row['returncode'], log

    def has_time():
        return now()+timedelta(seconds=690) < args.deadline

    try:
        arms = arms_for(args.experiment)
        bindings = {
            arm: training_input(argparse.Namespace(experiment=experiment, capacity_family=capacity,
                                                   arm=arm, training_audit=args.training_audit))
            for arm in arms
        }
        for arm in arms:
            _, binding, checkpoint, _ = bindings[arm]
            output = Path(f'{args.prefix}-{arm}-{suffix}.jsonl')
            meta = Path(str(output)+'.meta.json')
            audit = output.with_name(output.stem+'-audit.json')
            common = ['evaluate_ngram_placement.py', *family_args, '--arm', arm,
                      '--job',args.job,'--training-audit',args.training_audit,'--out',output]
            for attempt in range(1,16):
                old = load(meta) if meta.exists() else dict(status='new',items=0)
                if old['status']=='complete':
                    break
                if not has_time():
                    return 'time_budget'
                assert old['status'] in ('new','time_budget','memory_pressure')
                paths = [args.training_audit,checkpoint]
                if meta.exists():
                    paths.extend([output,meta])
                if old['status']=='memory_pressure' or not normal(memory()):
                    if not review(state,args.receipt,args.deadline,paths):
                        return 'resource_review_incomplete'
                command = [*common,'--deadline',args.deadline.isoformat(),'--seconds','900']
                if meta.exists():
                    command.append('--resume')
                call(f'{arm}-eval-{attempt:02}', command)
                new = load(meta)
                state['arms'][arm] = {key:new[key] for key in ('status','items','correct')}
                write_json(args.receipt, state)
                assert new['items']>old['items'] or not has_time(), 'No durable prediction progress'
            else:
                raise RuntimeError(f'Unexpected attempts for {arm}')
            if not audit.exists():
                if pressure()!=1 and not review(state,args.receipt,args.deadline,[output,meta,checkpoint]):
                    return 'resource_review_incomplete'
                command = [*common,'--audit',audit]
                result, log = call(f'{arm}-audit', command, check=False)
                if result:
                    assert not audit.exists() and 'assert pressure() == 1' in log.read_text(), log
                    if not review(state,args.receipt,args.deadline,[output,meta,checkpoint]):
                        return 'resource_review_incomplete'
                    call(f'{arm}-audit-after-review', command)
            receipt = load(audit)
            assert receipt['complete'] and receipt['identity']['checkpoint_sha256']==binding['checkpoint_sha256']
            assert receipt['identity']['training_audit_sha256']==state['training_audit_sha256']
            assert receipt['output_sha256']==digest(output)
            state['arms'][arm] = dict(status='complete',items=receipt['items'],correct=receipt['correct'],
                                      output_sha256=receipt['output_sha256'],audit_sha256=digest(audit))
            write_json(args.receipt,state)
        summary = Path(f'{args.prefix}-{args.job}-paired.json')
        if not summary.exists():
            call('paired',['summarize_ngram_quality.py', *family_args,
                           '--prefix',args.prefix,'--job',suffix,'--out',summary])
        paired = load(summary)
        assert paired['status']=='complete_paired_public_job' and list(paired['arms'])==arms
        assert paired['job'] == suffix
        assert paired['common_identity']['training_audit_sha256']==state['training_audit_sha256']
        for arm in arms:
            assert all(paired['arms'][arm][key]==state['arms'][arm][key]
                       for key in ('output_sha256','audit_sha256'))
        state.update(summary=str(summary),summary_sha256=digest(summary))
        return 'complete'
    except BaseException as exc:
        state.update(status='needs_diagnosis',error=repr(exc))
        raise
    finally:
        state['ended'] = now().isoformat()
        write_json(args.receipt,state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=tuple(load(SCOPE)['families']))
    parser.add_argument('--job', required=True)
    parser.add_argument('--training-audit', required=True, type=Path)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=datetime.fromisoformat)
    args = parser.parse_args()
    assert args.prefix.parent.resolve()==args.receipt.parent.resolve()==RUNS.resolve()
    assert not args.receipt.exists() and now() < args.deadline <= now()+timedelta(hours=2)
    status = execute(args)
    state = load(args.receipt)
    write_json(args.receipt,dict(state,status=status))


if __name__=='__main__':
    main()
