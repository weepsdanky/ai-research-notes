"""Serial bounded GPQA calls using one fixed source archive and round deadline."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error('Timezone required')
    output = 'runs/eval-paper-gpqa-full.jsonl'
    metadata = Path(output+'.meta.json')
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopped and datetime.now().astimezone() < deadline-timedelta(seconds=200):
        meta = json.loads(metadata.read_text()) if metadata.exists() else None
        if meta and meta['status'] == 'complete':
            break
        if meta and meta['status'] not in ('time_budget', 'interrupted_signal'):
            raise RuntimeError('Investigate previous status before resuming GPQA')
        index = 1
        while (log := Path(f'runs/paper-gpqa-{index:03d}.log')).exists():
            index += 1
        command = [sys.executable, 'run_with_deadline.py', '--deadline', args.deadline, '--', sys.executable,
                   'evaluate_batched.py', 'runs/qwen-sparse-paper/last.pt', '--suite', 'gpqa', '--tasks', 'gpqa-*',
                   '--out', output, '--full', '--tokens', '600', '--context', '8192', '--prefill-chunk', '128',
                   '--batch-size', '8', '--gdn-scan', 'chunk', '--prefix-cache', '--grouped-gqa', '--stop-pressure', '2', '--device', 'cpu']
        if meta:
            command.append('--resume')
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            state = dict(started=datetime.now().astimezone().isoformat(), deadline=args.deadline,
                         archive=args.archive, child_pid=child.pid, log=str(log), status='running')
            Path('runs/gpqa-runner-state.json').write_text(json.dumps(state, indent=2)+'\n')
            while child.poll() is None:
                if not stopped and datetime.now().astimezone() >= deadline-timedelta(seconds=150):
                    stop(signal.SIGINT, None)
                time.sleep(1)
            state.update(status='terminal', returncode=child.returncode, ended=datetime.now().astimezone().isoformat())
            Path('runs/gpqa-runner-state.json').write_text(json.dumps(state, indent=2)+'\n')
        if child.returncode:
            raise RuntimeError('GPQA child failed; no automatic restart')
        child = None
        subprocess.run([sys.executable, 'audit_gpqa_results.py', output, '--archive', args.archive,
                        '--allow-partial', '--out', 'runs/paper-gpqa-saved-audit.json'], check=True)
        meta = json.loads(metadata.read_text())
        if meta['status'] not in ('complete', 'time_budget', 'interrupted_signal'):
            break
    if metadata.exists() and json.loads(metadata.read_text())['status'] == 'complete':
        subprocess.run([sys.executable, 'audit_gpqa_results.py', output, '--archive', args.archive,
                        '--out', 'runs/paper-gpqa-completed-audit.json'], check=True)


if __name__ == '__main__':
    main()
