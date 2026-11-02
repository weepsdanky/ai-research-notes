"""Join Nsight UTC time to SSE phases and union overlapping GPU/API intervals."""
import argparse
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from analyze_phase_trace import merged


def union_seconds(spans, start, end):
    return sum(b - a for a, b in merged(spans, start, end)) / 1e9


def analyze(database, folder):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    pid = result['server_pid']
    entry = result['request']
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    connection.text_factory = lambda value: value.decode('utf-8', errors='replace')
    epoch = connection.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    details = [dict(row) for row in connection.execute('SELECT * FROM ANALYSIS_DETAILS')]
    if len(details) != 1:
        raise RuntimeError('This analyzer expects one capture')
    duration = details[0]['duration']
    start = result['request_started_unix_s'] * 1e9 - epoch
    first = start + entry['time_to_first_content_s'] * 1e9
    finish = start + entry['total_s'] * 1e9
    diagnostics = [dict(row) for row in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    warnings = [row for row in diagnostics if row['severity'] >= 2 or
                any(word in row['text'].lower() for word in ('missing', 'dropped')) or
                ('not supported' in row['text'].lower() and 'legacy (software instrumented)' not in row['text'])]
    driver_warning = any('Installed CUDA driver version' in row['text'] and 'not supported' in row['text']
                         for row in diagnostics)
    dropped = [int(match[1]) for row in diagnostics
               if (match := re.search(r'incomplete CUPTI events dropped:\s*(\d+)', row['text']))]
    # Leave out stop/start boundaries, where an incomplete activity may be dropped.
    phases = {'prefill_to_first_content': (max(start, .25e9), min(first, duration - .25e9)),
              'decode_after_first_content': (max(first, .25e9), min(finish, duration - .25e9))}
    phases = {name: (a, b) for name, (a, b) in phases.items() if b > a}
    spans = {}
    counts = {}
    copy_bytes = 0
    for kind in ('KERNEL', 'GRAPH_TRACE', 'MEMCPY', 'MEMSET'):
        rows = connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_' + kind).fetchall()
        rows = [row for row in rows if ((row['globalPid'] >> 24) & 0xffffff) == pid]
        spans[kind] = [(row['start'], row['end']) for row in rows]
        counts[kind] = len(rows)
        if kind == 'MEMCPY':
            copy_bytes = sum(row['bytes'] for row in rows)
    apis = connection.execute('''SELECT r.start,r.end,r.globalTid,r.correlationId,s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId''').fetchall()
    apis = [row for row in apis if ((row['globalTid'] >> 24) & 0xffffff) == pid]
    checks = {}
    for api_name, kind in (('cudaGraphLaunch', 'GRAPH_TRACE'), ('cudaStreamSynchronize', 'SYNCHRONIZATION')):
        ids = {row['correlationId'] for row in apis if row['name'].startswith(api_name)
               and row['start'] >= .25e9 and row['end'] <= duration - .25e9}
        # Ignore capture-specific host bits when selecting the target process.
        activity_ids = {row['correlationId'] for row in connection.execute(
            'SELECT correlationId,globalPid FROM CUPTI_ACTIVITY_KIND_' + kind)
            if ((row['globalPid'] >> 24) & 0xffffff) == pid}
        checks[api_name] = {'interior_unique_api_ids': len(ids), 'matched_activity_ids': len(ids & activity_ids),
                            'unmatched_api_ids': len(ids - activity_ids)}
    sync = [(row['start'], row['end']) for row in apis if 'Synchronize' in row['name']]
    output = {'server_pid': pid, 'capture_duration_s': duration / 1e9,
              'session_start_utc_epoch_ns': epoch, 'request_start_relative_ns': start,
              'request_first_content_relative_ns': first,
              'source_database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
              'activity_counts': counts, 'captured_memcpy_bytes': copy_bytes,
              'quality': 'unsupported-driver-exploratory' if driver_warning else
                         'completeness-warnings-require-review' if warnings else 'no-diagnostic-warnings',
              'cuda_driver_compatibility_warning': driver_warning,
              'known_incomplete_events_dropped': sum(dropped),
              'interior_correlation_checks': checks,
              'diagnostic_warnings': warnings, 'diagnostics': diagnostics, 'phases': {},
              'notes': ['Phases are intersected with the capture; first/last 250ms are excluded.',
                        'GPU intervals union kernels, entire CUDA graphs and copies without double counting.',
                        'A graph span may include internal gaps; this is activity coverage, not SM utilization.',
                        'Synchronize API duration includes host/driver overhead, not exclusively device execution.',
                        'Activity coverage is not a measured Amdahl critical-path fraction.',
                        'Graph tracing is coarse: direct-kernel rankings omit kernels inside recorded graphs.']}
    for name, (a, b) in phases.items():
        wall = (b - a) / 1e9
        gpu = union_seconds([span for values in spans.values() for span in values], a, b)
        synchronization = union_seconds(sync, a, b)
        api_s = collections.Counter()
        for row in apis:
            api_s[row['name']] += max(0, min(b, row['end']) - max(a, row['start'])) / 1e9
        output['phases'][name] = {
            'capture_start_relative_s': a / 1e9, 'capture_end_relative_s': b / 1e9,
            'wall_s': wall, 'gpu_activity_union_s': gpu, 'gpu_activity_coverage': gpu / wall,
            'synchronize_api_union_s': synchronization, 'synchronize_api_coverage': synchronization / wall,
            'activity_union_s_by_kind': {kind: union_seconds(values, a, b) for kind, values in spans.items()},
            'api_duration_sum_s_top': api_s.most_common(15)}
    (folder / 'cuda-analysis.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', nargs='?', type=Path)
    parser.add_argument('folder', nargs='?', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        assert union_seconds([(0, 2e9), (1e9, 3e9)], 0, 4e9) == 3
        assert union_seconds([(0, 2e9)], .5e9, 1.5e9) == 1
        assert ((281807769567232 >> 24) & 0xffffff) == 19836
        print('CUDA interval clipping/union and Nsight process-ID checks passed')
    else:
        if args.database is None or args.folder is None:
            parser.error('database and folder are required')
        output = analyze(args.database, args.folder)
        print(json.dumps({'quality': output['quality'], 'phases': output['phases']}, indent=2))
