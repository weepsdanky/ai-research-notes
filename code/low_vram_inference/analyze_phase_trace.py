"""Analyze target-only xperf CSV; time values are microseconds.

Hard-fault I/O intervals, scheduled CPU time and off-CPU waits overlap across
threads. They must not be added to produce a wall-clock pie chart.
"""
import argparse
import bisect
import collections
import csv
import json
import re
from pathlib import Path


def merged(intervals, start, end):
    result = []
    for left, right in sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def seconds(intervals, start, end):
    return sum(b - a for a, b in merged(intervals, start, end)) / 1e6


def intersect(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def analyze(folder, request_start_us):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    pid = result['server_pid']
    entry = result['request']
    first = request_start_us + entry['time_to_first_content_s'] * 1e6
    finish = request_start_us + entry['total_s'] * 1e6
    phases = {'prefill_to_first_content': (request_start_us, first), 'decode_after_first_content': (first, finish)}
    export_info = folder / 'export-info.json'
    trace_windows = [(request_start_us, finish)]
    if export_info.exists():
        windows = json.loads(export_info.read_text()).get('sample_windows_us')
        if windows:
            phases = windows
            trace_windows = sorted(windows.values())
    running = collections.defaultdict(list)
    waits = collections.defaultdict(list)
    ready = collections.defaultdict(list)
    faults, samples, switches = [], [], []
    schemas = {}
    active, pending = {}, {}
    seen_cpus = set()
    current_window = None
    window_starts = [pair[0] for pair in trace_windows]
    lost_switch_pairs = 0
    boundary_seeded_switches = 0

    def close_window(upper):
        for tid, left in active.values():
            running[tid].append((left, upper))
        for tid, (left, state, reason, became_ready) in pending.items():
            if state == 'Waiting':
                waits[(tid, reason)].append((left, became_ready if became_ready is not None else upper))
                if became_ready is not None:
                    ready[tid].append((became_ready, upper))
            else:
                ready[tid].append((left, upper))
        active.clear()
        pending.clear()
        seen_cpus.clear()
    source = folder / 'target-events.csv'
    with source.open(encoding='utf-8', errors='replace', newline='') as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            row = [x.strip() for x in row]
            if len(row) < 2:
                continue
            kind = row[0]
            if row[1] == 'TimeStamp':
                schemas[kind] = row
                continue
            if kind not in schemas or not row[1].isdigit():
                continue
            event = dict(zip(schemas[kind], row))
            at = int(row[1])
            window = bisect.bisect_right(window_starts, at) - 1
            if window < 0 or at > trace_windows[window][1]:
                continue
            if window != current_window:
                if current_window is not None:
                    close_window(trace_windows[current_window][1])
                current_window = window
            if kind == 'CSwitch':
                new = int(event['New TID']) if target_pid(event['New Process Name ( PID)']) == pid else None
                old = int(event['Old TID']) if target_pid(event['Old Process Name ( PID)']) == pid else None
                cpu = int(event['CPU'])
                if old is not None:
                    if cpu in active and active[cpu][0] == old:
                        running[old].append((active[cpu][1], at))
                    elif cpu not in seen_cpus:
                        # The first switch-out identifies the CPU's owner at the window start.
                        running[old].append((trace_windows[window][0], at))
                        boundary_seeded_switches += 1
                    else:
                        lost_switch_pairs += 1
                    pending[old] = (at, event['OldState'], event['Wait Reason'], None)
                    switches.append({'time_us': at, 'tid': old, 'state': event['OldState'], 'reason': event['Wait Reason']})
                active.pop(cpu, None)
                seen_cpus.add(cpu)
                if new is not None:
                    active[cpu] = (new, at)
                    if new in pending:
                        left, state, reason, became_ready = pending.pop(new)
                        if state == 'Waiting':
                            right = became_ready if became_ready is not None else at
                            waits[(new, reason)].append((left, right))
                            if became_ready is not None:
                                ready[new].append((became_ready, at))
                        else:
                            ready[new].append((left, at))
            elif kind == 'ReadyThread':
                if target_pid(event.get('Rdy Process Name ( PID)', '')) == pid:
                    tid = int(event['Rdy TID'])
                    if tid in pending and pending[tid][3] is None:
                        left, state, reason, _ = pending[tid]
                        pending[tid] = (left, state, reason, at)
            elif kind == 'HardFault' and target_pid(event['Process Name ( PID)']) == pid:
                faults.append({'start': at - int(event['ElapsedTime']), 'end': at,
                               'tid': int(event['ThreadID']), 'bytes': int(event['IOSize'], 0),
                               'file': event['FileName'].strip('"')})
            elif kind == 'SampledProfile' and target_pid(event['Process Name ( PID)']) == pid:
                samples.append((at, int(event['ThreadID']), event['Image!Function'], int(event.get('Count', '1'))))
    if current_window is not None:
        close_window(trace_windows[current_window][1])
    output = {'pid': pid, 'request_start_trace_us': request_start_us,
              'unmatched_first_or_lost_switch_outs': lost_switch_pairs,
              'boundary_seeded_switch_outs': boundary_seeded_switches,
              'notes': ['CPU seconds are summed across threads, not wall seconds.',
                        'A hard-fault union means at least one target fault I/O was in flight; it is not exclusively stalled wall time.',
                        'Scheduler wait reasons alone do not prove GPU waits; compare CUDA APIs and GPU activity.',
                        'Initial off-CPU states before a sampled window are unknown; correlated waits may omit a boundary fragment.',
                        'Sampled CPU addresses in ggml-cpu may include barriers/spinning; module totals alone are not arithmetic time.'],
              'phases': {}}
    for name, (start, end) in phases.items():
        per_thread = {str(tid): seconds(spans, start, end) for tid, spans in running.items()}
        reason_s = collections.Counter()
        thread_reason_s = collections.defaultdict(dict)
        for (tid, reason), spans in waits.items():
            value = seconds(spans, start, end)
            reason_s[reason] += value
            thread_reason_s[str(tid)][reason] = value
        relevant = [f for f in faults if f['end'] > start and f['start'] < end]
        fault_thread_s = collections.Counter()
        file_bytes = collections.Counter()
        for fault in relevant:
            fault_thread_s[str(fault['tid'])] += (min(end, fault['end']) - max(start, fault['start'])) / 1e6
            file_bytes[fault['file']] += fault['bytes']
        functions = collections.Counter()
        fault_union = merged([(f['start'], f['end']) for f in relevant], start, end)
        fault_starts = [a for a, b in fault_union]
        modules_during_fault = collections.Counter()
        modules_without_fault = collections.Counter()
        for at, tid, function, count in samples:
            if start <= at < end:
                functions[function] += count
                index = bisect.bisect_right(fault_starts, at) - 1
                destination = modules_during_fault if index >= 0 and at < fault_union[index][1] else modules_without_fault
                destination[function.split('!')[0]] += count
        modules = collections.Counter()
        for function, count in functions.items():
            modules[function.split('!')[0]] += count
        duration = (end - start) / 1e6
        correlated_waits = []
        correlated_thread_s = 0
        for tid in set(f['tid'] for f in relevant):
            fault_spans = merged([(f['start'], f['end']) for f in relevant if f['tid'] == tid], start, end)
            wait_spans = merged([span for (thread, reason), spans in waits.items() if thread == tid
                                 for span in spans], start, end)
            overlap = intersect(fault_spans, wait_spans)
            correlated_thread_s += seconds(overlap, start, end)
            correlated_waits.extend(overlap)
        output['phases'][name] = {
            'wall_s': duration, 'scheduled_cpu_s': sum(per_thread.values()),
            'average_running_logical_cpus': sum(per_thread.values()) / duration,
            'cpu_s_by_thread': dict(sorted(per_thread.items(), key=lambda item: -item[1])),
            'scheduler_wait_thread_s_by_reason': dict(reason_s),
            'scheduler_wait_thread_s_by_thread_reason': dict(thread_reason_s),
            'ready_queue_thread_s': sum(seconds(spans, start, end) for spans in ready.values()),
            'hard_fault_count': len(relevant),
            'hard_fault_bytes': sum(f['bytes'] for f in relevant),
            'hard_fault_io_thread_s': sum(fault_thread_s.values()),
            'hard_fault_io_any_thread_union_s': seconds([(f['start'], f['end']) for f in relevant], start, end),
            'hard_fault_io_s_by_thread': dict(fault_thread_s),
            'hard_fault_correlated_offcpu_thread_s': correlated_thread_s,
            'hard_fault_correlated_offcpu_any_thread_union_s': seconds(correlated_waits, start, end),
            'hard_fault_bytes_by_file': dict(file_bytes),
            'sample_counts_by_module': dict(modules),
            'sample_counts_by_module_during_any_fault_io': dict(modules_during_fault),
            'sample_counts_by_module_without_fault_io': dict(modules_without_fault),
            'sample_counts_top_addresses': functions.most_common(30),
            'switch_out_counts_by_state_reason': dict(collections.Counter(
                item['state'] + '/' + item['reason'] for item in switches
                if start <= item['time_us'] < end)),
        }
    (folder / 'etw-analysis.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


def target_pid(text):
    match = re.search(r'\(\s*(\d+)\)$', text)
    return int(match[1]) if match else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path)
    parser.add_argument('--request-start-us', type=int)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        assert merged([(0, 5), (3, 8), (8, 12), (20, 30)], 2, 25) == [[2, 12], [20, 25]]
        assert seconds([(0, 2e6), (1e6, 3e6)], 0, 4e6) == 3
        assert seconds([(0, 1)], 2, 3) == 0
        assert target_pid('llama-server.exe ( 123)') == 123
        assert target_pid('"Unknown" ( -1)') is None
        assert intersect([[0, 5], [8, 12]], [[3, 10]]) == [(3, 5), (8, 10)]
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / 'result.json').write_text(json.dumps({'server_pid': 123, 'request': {
                'time_to_first_content_s': .025, 'total_s': .04}}))
            (fixture / 'target-events.csv').write_text('''CSwitch,TimeStamp,New Process Name ( PID),New TID,Old Process Name ( PID),Old TID,OldState,Wait Reason,CPU
ReadyThread,TimeStamp,Rdy Process Name ( PID),Rdy TID
HardFault,TimeStamp,Process Name ( PID),ThreadID,ElapsedTime,IOSize,FileName
SampledProfile,TimeStamp,Process Name ( PID),ThreadID,Image!Function,Count
CSwitch,0,llama-server.exe (123),5,Idle (0),0,Ready,Executive,0
CSwitch,10000,Idle (0),0,llama-server.exe (123),5,Waiting,PageIn,0
HardFault,20000,llama-server.exe (123),5,10000,0x1000,model.gguf
ReadyThread,20000,llama-server.exe (123),5
CSwitch,22000,llama-server.exe (123),5,Idle (0),0,Ready,Executive,0
SampledProfile,23000,llama-server.exe (123),5,ggml-cpu.dll!compute,1
CSwitch,40000,Idle (0),0,llama-server.exe (123),5,Waiting,UserRequest,0
''')
            check = analyze(fixture, 0)['phases']
            assert abs(check['prefill_to_first_content']['scheduled_cpu_s'] - .013) < 1e-10
            assert abs(check['decode_after_first_content']['scheduled_cpu_s'] - .015) < 1e-10
            assert check['prefill_to_first_content']['hard_fault_io_any_thread_union_s'] == .01
            assert check['prefill_to_first_content']['ready_queue_thread_s'] == .002
            assert check['prefill_to_first_content']['hard_fault_bytes'] == 4096
            (fixture / 'export-info.json').write_text(json.dumps({'sample_windows_us': {
                'a': [0, 10000], 'b': [30000, 40000]}}))
            sparse = analyze(fixture, 0)['phases']
            assert sparse['a']['scheduled_cpu_s'] == .01
            assert sparse['b']['scheduled_cpu_s'] == .01
        print('Interval clipping, PID parsing and synthetic CPU/fault/ready timeline checks passed')
    else:
        if args.folder is None or args.request_start_us is None:
            parser.error('folder and --request-start-us are required')
        summary = analyze(args.folder, args.request_start_us)
        print(json.dumps({name: {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
                          for name, data in summary['phases'].items()}, indent=2))
