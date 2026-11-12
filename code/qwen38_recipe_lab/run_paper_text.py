"""Serial full paper-checkpoint text evaluation with native round deadlines.

No datasets, token budgets, prompts, model sources or prior predictions change.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

PLAN = Path('profiles/paper_text_queue.json')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error('Timezone required')
    plan = json.loads(PLAN.read_text())
    checkpoint, sha = plan['checkpoint'], plan['checkpoint_sha256']
    assert digest(checkpoint) == sha
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def invoke(command, log):
        nonlocal child
        print(datetime.now().astimezone().isoformat(), log, flush=True)
        with Path(log).open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            state = dict(deadline=args.deadline, job=job, pid=child.pid, log=log, started=datetime.now().astimezone().isoformat(), status='child_running', driver_sha256=digest(__file__))
            Path('runs/paper-text-runner-state.tmp').write_text(json.dumps(state, indent=2)+'\n')
            Path('runs/paper-text-runner-state.tmp').replace('runs/paper-text-runner-state.json')
            tick = time.monotonic()
            while child.poll() is None:
                if not stopping and datetime.now().astimezone() >= deadline-timedelta(seconds=150):
                    stop(signal.SIGINT, None)
                if time.monotonic()-tick >= 45:
                    print(datetime.now().astimezone().isoformat(), 'live_pid', child.pid, flush=True)
                    tick = time.monotonic()
                time.sleep(1)
            code = child.returncode
            state.update(ended=datetime.now().astimezone().isoformat(), status='child_terminal', returncode=code)
            Path('runs/paper-text-runner-state.tmp').write_text(json.dumps(state, indent=2)+'\n')
            Path('runs/paper-text-runner-state.tmp').replace('runs/paper-text-runner-state.json')
        child = None
        if code:
            raise RuntimeError(f'Child exit {code}; inspect {log}, no automatic restart')

    def log_path(label):
        index = 1
        while (path := Path(f'runs/paper-text-{label}-{index:03d}.log')).exists():
            index += 1
        return str(path)

    for job in plan['jobs']:
        name, tasks, suite = job['name'], job['tasks'], job['suite']
        output = f'runs/eval-paper-{name}-full.jsonl'
        meta_path = Path(output+'.meta.json')
        while not stopping and datetime.now().astimezone() < deadline-timedelta(seconds=200):
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            if meta is not None:
                assert meta['identity']['checkpoint_sha256'] == sha
                assert sum(t['total_dataset'] for t in meta['tasks'].values()) == job['total_items'], 'Full dataset target drift'
                if meta['status'] == 'running':
                    raise RuntimeError(f'Existing running state: {output}; verify its actual process first')
                if meta['status'] not in ('complete', 'time_budget', 'interrupted_signal'):
                    print('Resource/error stop:', meta['status'], flush=True)
                    return
                if meta['full_dataset_generated']:
                    audit = Path(f'runs/paper-{name}-completed-audit.json')
                    if audit.exists():
                        verified = json.loads(audit.read_text())['files'][output]
                        assert verified['full_dataset_generated'] and verified['file_sha256'] == digest(output)
                    else:
                        invoke([sys.executable, 'audit_core_results.py', output, '--suite', suite, '--out', str(audit)], log_path(name+'-audit'))
                    print('Audited full dataset:', name, flush=True)
                    break
            command = [sys.executable, 'run_with_deadline.py', '--deadline', args.deadline, '--',
                       sys.executable, 'evaluate_batched.py', checkpoint, '--suite', suite, '--tasks', *tasks,
                       '--out', output, '--full', '--tokens', '64', '--context', str(job['context']), '--prefill-chunk', '128',
                       '--batch-size', '8', '--gdn-scan', 'chunk', '--prefix-cache', '--grouped-gqa',
                       '--stop-pressure', '4', '--device', 'cpu']
            if meta is not None:
                command.append('--resume')
            invoke(command, log_path(name))
        else:
            return
    print('All declared public text collections audited; code and gated/private requirements remain separate.', flush=True)


if __name__ == '__main__':
    main()
