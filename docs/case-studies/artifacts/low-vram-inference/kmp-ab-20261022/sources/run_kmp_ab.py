"""Three paired, order-alternating KMP_BLOCKTIME trials; no profiler, stdlib only."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def cpu_at(samples, stamp):
    """Linear interpolation of cumulative process user+kernel CPU seconds."""
    for left, right in zip(samples, samples[1:]):
        if left['unix_s'] <= stamp <= right['unix_s']:
            fraction = (stamp - left['unix_s']) / (right['unix_s'] - left['unix_s'])
            def value(sample):
                return sample['process_cpu_user_seconds'] + sample['process_cpu_kernel_seconds']
            return value(left) + fraction * (value(right) - value(left))
    raise ValueError('CPU samples do not bracket request boundary: ' + str(stamp))


def summarize(folder):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'request-passed':
        raise RuntimeError('Request failed: ' + str(folder))
    entry = result['request']
    count = entry['usage']['completion_tokens']
    if count != 64 or not entry['text'].strip():
        raise RuntimeError('Expected 64 output tokens')
    samples = [json.loads(line) for line in (folder / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if any('process_cpu_user_seconds' not in s or 'process_cpu_kernel_seconds' not in s for s in samples):
        raise RuntimeError('Missing CPU counters')
    start = result['request_started_unix_s']
    # request() uses perf_counter internally; the adjacent caller UTC stamp is the join anchor.
    first = start + entry['time_to_first_content_s']
    end = start + entry['total_s']
    cpus = result['logical_processors']
    row = {'folder': str(folder), 'server_pid': result['server_pid'],
           'openmp_environment': result['openmp_environment'], 'server_command': result['server_command'],
           'output_sha256': hashlib.sha256(entry['text'].encode()).hexdigest(),
           'prompt_tokens': entry['usage']['prompt_tokens'], 'completion_tokens': count,
           'ttft_s': entry['time_to_first_content_s'], 'total_s': entry['total_s'],
           'decode_tokens_per_s': (count - 1) / (end - first),
           'request_anchor_gap_s': result['request_ended_unix_s'] - end,
           'sample_count': len(samples),
           'max_sample_gap_s': max(b['unix_s'] - a['unix_s'] for a, b in zip(samples, samples[1:])),
           'logical_processors': cpus}
    for name, left, right in [('request', start, end), ('prefill', start, first), ('decode', first, end)]:
        cpu = cpu_at(samples, right) - cpu_at(samples, left)
        row[name + '_cpu_seconds'] = cpu
        row[name + '_mean_cpu_cores'] = cpu / (right - left)
        row[name + '_cpu_percent_machine'] = 100 * cpu / (right - left) / cpus
        row[name + '_cpu_seconds_per_output_token'] = cpu / (count - 1 if name == 'decode' else count)
    inside = [s for s in samples if start <= s['unix_s'] <= end]
    row['page_fault_count_delta_sampled'] = inside[-1]['page_fault_count'] - inside[0]['page_fault_count']
    row['page_fault_note'] = 'Includes soft faults; sampled interior excludes boundary fragments. Not hard-fault I/O.'
    return row


def compare(a, b):
    return {metric: 100 * (b[metric] / a[metric] - 1) for metric in (
        'decode_tokens_per_s', 'ttft_s', 'request_cpu_percent_machine',
        'decode_cpu_percent_machine', 'request_cpu_seconds', 'decode_cpu_seconds_per_output_token')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(r'D:\ai-research\low-vram-inference'))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        fixture = [dict(unix_s=x, process_cpu_user_seconds=2*x,
                        process_cpu_kernel_seconds=x) for x in (0, 1, 3)]
        assert cpu_at(fixture, .5) == 1.5
        assert cpu_at(fixture, 3) == 9
        assert math.isclose((cpu_at(fixture, 2) - cpu_at(fixture, .5)) / 1.5, 3)
        for boundary in (-1, 4):
            try:
                cpu_at(fixture, boundary)
            except ValueError:
                pass
            else:
                raise AssertionError('Unbracketed CPU boundary accepted')
        print('CPU interpolation/coverage self-check passed')
        return
    if os.name != 'nt':
        raise RuntimeError('Run on the Windows target')
    folder = args.root / 'results' / ('kmp-ab-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)
    collector = Path(__file__).with_name('profile_phases.py')
    batch = {'status': 'running', 'started_unix_s': time.time(),
             'conditions': {'A': 'KMP_BLOCKTIME unset', 'B': 'KMP_BLOCKTIME=0'},
             'schedule': [['A', 'B'], ['B', 'A'], ['A', 'B']],
             'parent_kmp_blocktime': os.environ.get('KMP_BLOCKTIME'),
             'method': 'Fresh server each request, OS cache uncontrolled, no profiler; 64 output tokens.',
             'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                 Path(__file__), collector, collector.with_name('run_deepseek.py'))},
             'runs': [], 'pairs': []}
    def save():
        (folder / 'batch.json').write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    print(str(folder), flush=True)
    try:
        for pair, order in enumerate(batch['schedule'], 1):
            rows = {}
            for condition in order:
                command = [sys.executable, str(collector), '--root', str(args.root), '--mode', 'control',
                           '--tokens', '64', '--kmp-blocktime', 'unset' if condition == 'A' else '0']
                print(f'Pair {pair} {condition}: starting', flush=True)
                # collector has bounded readiness/request deadlines and owns its server cleanup.
                call = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=720)
                (folder / f'pair-{pair}-{condition}.log').write_text(call.stdout + call.stderr, encoding='utf-8')
                if call.returncode:
                    raise RuntimeError(f'Collector failed for pair {pair} {condition}: ' + call.stdout + call.stderr)
                run_folder = Path(call.stdout.splitlines()[0].strip())
                row = summarize(run_folder)
                row.update(pair=pair, condition=condition, run_order=len(batch['runs']) + 1)
                expected = None if condition == 'A' else '0'
                if row['openmp_environment']['KMP_BLOCKTIME'] != expected:
                    raise RuntimeError('Child environment does not match condition')
                if batch['runs']:
                    previous = batch['runs'][0]
                    if row['output_sha256'] != previous['output_sha256'] or row['server_command'] != previous['server_command']:
                        raise RuntimeError('Output/command differs across conditions')
                    other_env = lambda r: {k: v for k, v in r['openmp_environment'].items() if k != 'KMP_BLOCKTIME'}
                    if other_env(row) != other_env(previous):
                        raise RuntimeError('Other OpenMP variables changed')
                rows[condition] = row
                batch['runs'].append(row)
                save()
                print(f'Pair {pair} {condition}: {row["decode_tokens_per_s"]:.4f} tok/s, '
                      f'TTFT {row["ttft_s"]:.3f}s, CPU {row["request_cpu_percent_machine"]:.2f}%', flush=True)
                time.sleep(5)
            batch['pairs'].append({'pair': pair, 'order': order, 'B_vs_A_percent': compare(rows['A'], rows['B'])})
            save()
        metrics = ['decode_tokens_per_s', 'ttft_s', 'request_cpu_percent_machine',
                   'decode_cpu_percent_machine', 'request_cpu_seconds', 'decode_cpu_seconds_per_output_token']
        batch['condition_summary'] = {
            condition: {key: {'median': statistics.median(r[key] for r in batch['runs'] if r['condition'] == condition),
                              'min': min(r[key] for r in batch['runs'] if r['condition'] == condition),
                              'max': max(r[key] for r in batch['runs'] if r['condition'] == condition)} for key in metrics}
            for condition in ('A', 'B')}
        batch['paired_median_percent'] = {key: statistics.median(p['B_vs_A_percent'][key] for p in batch['pairs'])
                                          for key in metrics}
        batch['status'] = 'six-requests-and-comparison-gates-passed'
    except Exception as error:
        batch['status'], batch['error'] = 'failed', repr(error)
        raise
    finally:
        batch['ended_unix_s'] = time.time()
        save()
    print(json.dumps(batch['paired_median_percent']), flush=True)


if __name__ == '__main__':
    main()
