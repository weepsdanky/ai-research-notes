"""Finish the original full fourteen-language HF-checkpoint MMMLU target serially."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from audit_core_results import digest

CHECKPOINT = 'runs/qwen-sparse/last.pt'
CHECKPOINT_SHA = '996f1abf6540637b63da742077db4bd7d49f2f13bbe247f8aca2710c3a072cb8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--controller-state', help='Round-owned monitor target file')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error('Timezone required')
    if datetime.now().astimezone() >= deadline-timedelta(seconds=200):
        print('No child launched: round cleanup reserve reached', flush=True)
        return
    assert digest(CHECKPOINT) == CHECKPOINT_SHA
    plan = json.loads(Path('profiles/mmmlu_eval.json').read_text())
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.controller_state:
        Path(args.controller_state).write_text(json.dumps(dict(pid=os.getpid(), expected_command='run_hf_mmmlu.py --deadline '+args.deadline, deadline=args.deadline))+'\n')

    def invoke(command, label):
        nonlocal child
        index = 1
        while (log := Path(f'runs/hf-mmmlu-{label}-{index:03d}.log')).exists():
            index += 1
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            state = dict(started=datetime.now().astimezone().isoformat(), deadline=args.deadline, language=lang,
                         pid=child.pid, driver_pid=os.getpid(), log=str(log), status='child_running', driver_sha256=digest(__file__))
            Path('runs/hf-mmmlu-runner-state.json').write_text(json.dumps(state, indent=2)+'\n')
            print(state['started'], str(log), child.pid, flush=True)
            while child.poll() is None:
                if not stopping and datetime.now().astimezone() >= deadline-timedelta(seconds=150):
                    stop(signal.SIGINT, None)
                time.sleep(1)
            state.update(ended=datetime.now().astimezone().isoformat(), status='child_terminal', returncode=child.returncode)
            Path('runs/hf-mmmlu-runner-state.json').write_text(json.dumps(state, indent=2)+'\n')
        if child.returncode:
            raise RuntimeError('Child failed; inspect saved state before any restart')
        child = None

    for lang in plan['languages_in_order']:
        output = f'runs/eval-sparse-mmmlu-{lang}-full.jsonl'
        meta_path = Path(output+'.meta.json')
        while not stopping and datetime.now().astimezone() < deadline-timedelta(seconds=200):
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            if meta:
                assert meta['identity']['checkpoint_sha256'] == CHECKPOINT_SHA
                assert set(meta['tasks']) == {'mmmlu-'+lang}
                assert sum(task['total_dataset'] for task in meta['tasks'].values()) == plan['items_per_language']
                if meta['status'] not in ('complete', 'time_budget', 'interrupted_signal'):
                    print('Investigate previous running/resource/error state:', output, meta['status'], flush=True)
                    return
                if meta['full_dataset_generated']:
                    # Historical languages use short audit filenames. Bind by prediction path.
                    matches = [(path, json.loads(path.read_text())['files'].get(output))
                               for path in Path('runs').glob('mmmlu-*-completed-audit.json')]
                    verified = next((item for _, item in matches if item is not None), None)
                    if verified is None:
                        invoke([sys.executable, 'audit_core_results.py', output, '--suite', 'mmmlu', '--out', f'runs/mmmlu-{lang}-completed-audit.json'], lang+'-audit')
                    else:
                        assert verified['full_dataset_generated'] and verified['items'] == 14042
                        assert verified['file_sha256'] == digest(output)
                    print('Audited full HF language:', lang, flush=True)
                    break
            command = [sys.executable, 'run_with_deadline.py', '--deadline', args.deadline, '--', sys.executable,
                       'evaluate_batched.py', CHECKPOINT, '--suite', 'mmmlu', '--tasks', 'mmmlu-'+lang,
                       '--out', output, '--full', '--tokens', '64', '--context', '8192', '--batch-size', '8',
                       '--prefill-chunk', '128', '--gdn-scan', 'chunk', '--prefix-cache', '--grouped-gqa',
                       '--stop-pressure', '4', '--device', 'cpu']
            if meta:
                command.append('--resume')
            invoke(command, lang)
        else:
            return
    print('All original HF MMMLU languages audited; paper predictions remain separate.', flush=True)


if __name__ == '__main__':
    main()
