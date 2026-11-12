"""Continue the declared Table 1 generation jobs using the frozen guard."""
import argparse
from datetime import datetime, timedelta
from itertools import count
import os
from pathlib import Path
import subprocess
import sys

from qwen35_quality_data import PROTOCOL, digest, load, training_binding, write_json
from run_control_quality import now
from run_qwen35_quality_guarded import prior_identity
from summarize_qwen35_quality import RUNS, generation_evidence


def pending(prior):
    if prior['status'] == 'saved_and_audited':
        assert prior['returncode'] == 0 and prior['native_status'] == 'time_budget'
        return 'resume'
    assert prior['status'] == 'stop_requires_review' and prior['returncode'] == 0
    signals = prior['signals']
    if prior['native_status'] == 'interrupted_signal' and signals and all(
        row['signal'] == 'SIGINT' and row['reason'] in ('deadline', 'wall_budget') and row['pressure'] == 1
        for row in signals
    ):
        return 'resume'
    assert prior['native_status'] in ('memory_pressure', 'interrupted_signal')
    assert all(row['signal'] == 'SIGINT' and row['reason'] == 'memory_pressure' for row in signals)
    assert prior['native_status'] == 'memory_pressure' or signals
    return 'review'


def plan(binding, jobs, states):
    arm = binding['arm']
    protocol = load(PROTOCOL)
    checked = {(binding['training_profile'], binding['training_audit'], binding['checkpoint'], arm): binding}
    result = {}
    for name, job in jobs.items():
        output = Path(job['output_template'].format(arm=arm))
        meta = Path(f'{output}.meta.json')
        assert output.exists() == meta.exists()
        entry = dict(output=str(output), action='start', items=0, target=job['target_items'])
        if output.exists():
            if name not in states:
                raise ValueError(f'{name} has saved predictions and requires their audited guard state')
            prior_path = states[name]
            prior = load(prior_path)
            assert prior['binding'] == binding and prior['job'] == name and prior['output'] == str(output)
            assert prior['predictions_sha256'] == digest(output) and prior['meta_sha256'] == digest(meta)
            audit = Path(prior['audit_file'])
            assert prior['audit_sha256'] == digest(audit)
            receipt = load(audit)
            snapshot = load(meta)
            assert receipt['status'] == 'passed_source_output_audit' and receipt['identity'] == snapshot['identity']
            assert receipt['identity']['binding'] == binding and receipt['identity']['job_spec'] == job
            assert receipt['items'] == prior['items'] == snapshot['completed_items']
            assert receipt['target'] == snapshot['target_items'] == entry['target']
            assert receipt['meta_sha256'] == prior['meta_sha256'] and receipt['predictions_sha256'] == prior['predictions_sha256']
            complete = receipt['items'] == entry['target']
            assert snapshot['status'] == prior['native_status']
            assert receipt['full_dataset_generated'] == prior['full_dataset_generated'] == snapshot['full_dataset_generated'] == complete
            if complete:
                generation_evidence(audit, protocol, jobs, {}, checked)
            entry.update(action='skip_complete' if complete else pending(prior), items=receipt['items'],
                         prior_state=str(prior_path), prior_state_sha256=digest(prior_path),
                         audit=str(audit), audit_sha256=prior['audit_sha256'])
        else:
            assert name not in states
        result[name] = entry
    return result


def run(args, binding, jobs, state, previous):
    arm = binding['arm']

    def call(name, attempt, extra, review=False):
        prefix = args.receipt.with_name(f'{args.receipt.stem}-{name}-{attempt:02}' + ('-review' if review else ''))
        path = Path(f'{prefix}-state.json')
        command = [sys.executable, 'run_qwen35_quality_guarded.py', '--arm', arm, '--job', name,
                   '--training-profile', args.training_profile, '--checkpoint', args.checkpoint,
                   '--audit', args.audit, '--deadline', args.deadline.isoformat(), '--prefix', str(prefix),
                   '--seconds', str(args.seconds), *extra]
        row = dict(command=command, state=str(path), started=now().isoformat())
        state['calls'].append(row)
        with Path(f'{prefix}-dispatch.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            write_json(args.receipt, state)
            row['returncode'] = child.wait()
        row['ended'] = now().isoformat()
        assert row['returncode'] == 0
        result = load(path)
        row['state_sha256'] = digest(path)
        state['results'][name] = result
        write_json(args.receipt, state)
        return path, result

    for name in jobs:
        for attempt in count(1):
            selected = {name: previous[name]} if name in previous else {}
            entry = plan(binding, {name: jobs[name]}, selected)[name]
            if entry['action'] == 'skip_complete':
                break
            if now() + timedelta(seconds=690) >= args.deadline:
                return 'time_budget'
            extra = []
            if entry['action'] in ('resume', 'review'):
                path = entry['prior_state']
                prior_identity(path, args.training_profile, args.audit, args.checkpoint, arm, name, args.deadline)
                extra = ['--resume', '--prior-state', path]
            if entry['action'] == 'review':
                _, result = call(name, attempt, [*extra, '--prepare-review'], review=True)
                if result['status'] != 'resource_review_created_no_model_child':
                    return result['status']
                if now() + timedelta(seconds=690) >= args.deadline:
                    return 'time_budget'
                extra += ['--review-stop', result['review_file']]
            path, result = call(name, attempt, extra)
            if result['status'] not in ('saved_and_audited', 'stop_requires_review'):
                return result['status']
            assert result['items'] > entry['items']
            previous[name] = path
    return 'generation_complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arm', 'training-profile', 'checkpoint', 'audit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--jobs', nargs='+')
    parser.add_argument('--prior-state', type=Path, nargs='+', default=[])
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--deadline', type=datetime.fromisoformat)
    args = parser.parse_args()
    assert args.receipt.parent.resolve() == RUNS.resolve() and not args.receipt.exists()
    assert 30 <= args.seconds <= 1140
    binding = training_binding(args.training_profile, args.audit, args.checkpoint, args.arm)
    declared = {job['name']: job for job in load(PROTOCOL)['jobs']}
    names = args.jobs or list(declared)
    assert len(names) == len(set(names))
    jobs = {name: declared[name] for name in names}
    previous = {}
    for path in args.prior_state:
        name = load(path)['job']
        assert name in jobs and name not in previous
        previous[name] = path
    initial = plan(binding, jobs, previous)
    state = dict(pid=os.getpid(), binding=binding, protocol_sha256=digest(PROTOCOL),
                 initial=initial, calls=[], results={})
    if args.prepare_only:
        write_json(args.receipt, dict(state, status='verified_generation_queue_inputs'))
        return
    assert args.deadline.tzinfo and now() + timedelta(seconds=690) < args.deadline <= now() + timedelta(hours=2)
    state.update(status='running', deadline=args.deadline.isoformat())
    write_json(args.receipt, state)
    try:
        state['status'] = run(args, binding, jobs, state, previous)
    except BaseException as exc:
        state.update(status='needs_diagnosis', error=repr(exc))
        raise
    finally:
        state['ended'] = now().isoformat()
        write_json(args.receipt, state)


if __name__ == '__main__':
    main()
