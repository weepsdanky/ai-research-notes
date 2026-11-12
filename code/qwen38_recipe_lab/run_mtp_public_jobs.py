"""Run and audit the declared full public MTP acceptance jobs on the Mac."""
import argparse
from datetime import datetime, timedelta
from itertools import count
import os
from pathlib import Path
import subprocess
import sys

from mtp_public_data import PROFILE
from qwen35_quality_data import digest, load, write_json
from run_control_quality import memory, normal, now, review
from run_dense_text_guarded import pressure

DEFAULT_JOBS = ('humaneval', 'mbpp', 'gsm8k', 'math')
FULL_JOBS = (*DEFAULT_JOBS, 'mt-bench')


def binding(args):
    profile = load(PROFILE)
    counts = {job: spec['items'] for job, spec in profile['jobs'].items()}
    additional = []
    if 'mt-bench' in args.jobs:
        from audit_mtp_coverage import MT_PROFILE

        mt = load(MT_PROFILE)
        counts['mt-bench'] = mt['conversation_count'] * mt['turns_per_conversation']
        additional.append(MT_PROFILE)
    if args.attention == 'full':
        from evaluate_mtp_attention import TRAIN_PROFILE, training_binding

        training, checkpoint, state = training_binding()
        assert args.checkpoint.resolve() == checkpoint.resolve()
        assert args.verification == 'serial'
        command = ['evaluate_mtp_attention.py']
        identity = dict(attention='full', training_state_sha256=digest(state))
        additional += [TRAIN_PROFILE, state]
    else:
        training = load(profile['draft_training_profile'])
        command = ['evaluate_mtp_public.py', '--checkpoint', args.checkpoint,
                   '--verification', args.verification]
        identity = {}
    assert all(job in counts for job in args.jobs)
    bound = [args.checkpoint, Path(training['target']), PROFILE, *additional]
    identity.update(verification=args.verification, profile_sha256=digest(PROFILE),
                    target_checkpoint_sha256=digest(training['target']),
                    draft_checkpoint_sha256=digest(args.checkpoint))
    return counts, command, bound, identity


def run(args):
    counts, worker, bound, expected_identity = binding(args)
    state = dict(status='running',pid=os.getpid(),deadline=args.deadline.isoformat(),
                 checkpoint_sha256=expected_identity['draft_checkpoint_sha256'],
                 attention=args.attention, verification=args.verification,
                 jobs=args.jobs,calls=[],reviews=[],results={})

    def save():
        write_json(args.receipt,state)

    def enough():
        return now()+timedelta(seconds=690)<args.deadline

    def call(label, command):
        log = args.receipt.with_name(f'{args.receipt.stem}-{label}.log')
        row = dict(label=label, command=[str(x) for x in command], log=str(log), started=now().isoformat())
        state['calls'].append(row)
        with log.open('x') as stream:
            process = subprocess.Popen([sys.executable,*row['command']],stdout=stream,stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            save()
            row['returncode'] = process.wait()
        row['ended'] = now().isoformat()
        save()
        assert row['returncode']==0, log
        return row

    save()
    try:
        for job in args.jobs:
            output = Path(f'{args.prefix}-{job}.jsonl')
            meta = Path(str(output)+'.meta.json')
            audit = output.with_name(output.stem+'-audit.json')
            job_worker, job_bound = worker, bound
            job_identity = dict(expected_identity, job=job)
            if args.attention == 'qsa' and job == 'mt-bench':
                from evaluate_mtp_attention import training_binding

                _, checkpoint, training_state = training_binding('qsa')
                assert checkpoint.resolve() == args.checkpoint.resolve()
                job_worker = ['evaluate_mtp_attention.py', '--attention', 'qsa']
                job_bound = [*bound, training_state]
                job_identity.update(attention='qsa', training_state_sha256=digest(training_state))
            command = [*job_worker, '--job', job, '--out', output]
            time_stop = None
            for attempt in count(1):
                if time_stop:
                    assert digest(output) == time_stop['output_sha256'], output
                    assert digest(meta) == time_stop['metadata_sha256'], meta
                old = load(meta) if meta.exists() else dict(status='new',totals=dict(items=0))
                assert meta.exists()==output.exists()
                if old['status']=='complete':
                    break
                if not enough():
                    state['status']='time_budget'
                    if meta.exists() and old['status'] in ('time_budget','memory_pressure') and pressure()==1:
                        partial = args.receipt.with_name(f'{args.receipt.stem}-{job}-partial-audit.json')
                        call(job+'-partial-audit',[*command,'--audit-out',partial])
                        checked = load(partial)
                        assert not checked['complete'] and checked['totals']['items']==old['totals']['items']
                        assert checked['target'] == counts[job]
                        assert all(checked['identity'][key] == value for key, value in job_identity.items())
                        assert checked['output_sha256']==digest(output)
                        state['results'][job] = dict(status=old['status'],items=checked['totals']['items'],
                                                     target=checked['target'],audit=str(partial),audit_sha256=digest(partial))
                    return
                assert old['status'] in ('new','running','time_budget','memory_pressure')
                if old['status'] in ('memory_pressure','running') or not normal(memory()):
                    paths = [*job_bound,output,meta] if meta.exists() else job_bound
                    if not review(state,args.receipt,args.deadline,paths):
                        state['status']='resource_review_incomplete'
                        return
                generate = [*command,'--deadline',args.deadline.isoformat(),'--seconds','900']
                if meta.exists():
                    generate.append('--resume')
                attempt_row = call(f'{job}-eval-{attempt:02}', generate)
                current = load(meta)
                assert current['status'] in ('complete', 'time_budget', 'memory_pressure')
                assert current['target'] == counts[job]
                assert all(current['identity'][key] == value for key, value in job_identity.items())
                assert current['totals']['items'] >= old['totals']['items']
                # A pressure stop during startup can happen before the first new row.
                # The next attempt must pass the usual hash-bound resource review.
                advanced = current['totals']['items'] > old['totals']['items']
                assert advanced or current['status'] == 'memory_pressure' or not enough()
                time_stop = None
                if current['status'] == 'time_budget':
                    time_stop = dict(items=current['totals']['items'], output=str(output),
                                     output_sha256=digest(output), metadata_sha256=digest(meta))
                    attempt_row['time_stop'] = time_stop
                state['results'][job] = dict(status=current['status'],items=current['totals']['items'],
                                             target=counts[job],output=str(output))
                save()
            if not audit.exists():
                if pressure()!=1 and not review(state,args.receipt,args.deadline,[*job_bound,output,meta]):
                    state['status']='resource_review_incomplete'
                    return
                call(job+'-audit',[*command,'--audit-out',audit])
            verified = load(audit)
            assert verified['complete'] and verified['target'] == counts[job] == verified['totals']['items']
            identity = verified['identity']
            assert all(identity[key] == value for key, value in job_identity.items())
            assert verified['output_sha256']==digest(output)
            state['results'][job] = dict(status='complete',items=verified['target'],audit=str(audit),
                                         audit_sha256=digest(audit),acceptance=verified['acceptance'])
            save()
        state['status']='complete'
    except BaseException as exc:
        state.update(status='needs_diagnosis',error=repr(exc))
        raise
    finally:
        state['ended']=now().isoformat()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', choices=FULL_JOBS)
    parser.add_argument('--attention', choices=('qsa', 'full'), default='qsa')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=datetime.fromisoformat)
    parser.add_argument('--verification', choices=('batched', 'serial'), default='batched')
    args = parser.parse_args()
    if args.jobs is None:
        args.jobs = FULL_JOBS if args.attention == 'full' else DEFAULT_JOBS
    if args.attention == 'full' and args.verification != 'serial':
        parser.error('Full Attention requires --verification serial')
    if 'mt-bench' in args.jobs and args.verification != 'serial':
        parser.error('MT-Bench requires --verification serial')
    assert len(set(args.jobs))==len(args.jobs)
    assert args.prefix.parent.resolve()==args.receipt.parent.resolve()==Path('runs').resolve()
    assert not args.receipt.exists() and now()<args.deadline<=now()+timedelta(hours=2)
    run(args)


if __name__=='__main__':
    main()
