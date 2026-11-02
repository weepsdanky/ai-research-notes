"""Export only target CPU/scheduler/hard-fault events from an owned Windows ETL."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

from analyze_phase_trace import target_pid
from profile_phases import XPERF


def export(folder):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    pid = result['server_pid']
    etl = folder / 'cpu-faults.etl'
    for name, action in [('trace-stats', 'tracestats'), ('marks', 'marks')]:
        call = subprocess.run([str(XPERF), '-i', str(etl), '-a', action],
                              capture_output=True, text=True, errors='replace', timeout=180)
        (folder / (name + '.txt')).write_text(call.stdout + call.stderr, encoding='utf-8')
        if call.returncode:
            raise RuntimeError(action + ' failed')
    stats = (folder / 'trace-stats.txt').read_text()
    match = re.search(r'Start time \(UTC\)\s*:\s*(\d+/\d+/\d+:\d+:\d+:\d+\.\d+)', stats)
    if not match:
        raise RuntimeError('Missing trace UTC start')
    stamp = match[1]
    seconds, fraction = stamp.split('.')
    began = datetime.datetime.strptime(seconds, '%Y/%m/%d:%H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
    began += float('0.' + fraction)
    start_us = round((result['request_started_unix_s'] - began) * 1e6)
    end_us = start_us + round(result['request']['total_s'] * 1e6)
    kept = {'CSwitch', 'ReadyThread', 'HardFault', 'SampledProfile', 'Image',
            'Image/Load', 'Image/DCStart', 'Image/DCStop'}
    first_us = start_us + round(result['request']['time_to_first_content_s'] * 1e6)
    windows = {'prefill_5s': [start_us + 5000000, start_us + 6000000]}
    for offset in (5, 25, 45):
        left = first_us + offset * 1000000
        if left + 1000000 < end_us:
            windows['decode_' + str(offset) + 's'] = [left, left + 1000000]
    schemas, counts = {}, {}
    (folder / 'target-events.csv').write_text('', encoding='utf-8')
    with (folder / 'dumper-errors.txt').open('w', encoding='utf-8') as errors:
        for window, (left, right) in windows.items():
            print('Exporting ' + window, flush=True)
            process = subprocess.Popen([str(XPERF), '-i', str(etl), '-a', 'dumper',
                                        '-range', str(left), str(right)],
                                       stdout=subprocess.PIPE, stderr=errors, text=True, errors='replace')
            with (folder / 'target-events.csv').open('a', encoding='utf-8', newline='') as output:
                writer = csv.writer(output)
                for line in process.stdout:
                    kind = line.lstrip().split(',', 1)[0]
                    if kind not in kept:
                        continue
                    row = [value.strip() for value in next(csv.reader(io.StringIO(line), skipinitialspace=True))]
                    if len(row) < 2:
                        continue
                    if row[1] == 'TimeStamp':
                        schemas[kind] = row
                        writer.writerow(row)
                        continue
                    if kind not in schemas:
                        continue
                    schema = schemas[kind]
                    event = dict(zip(schema, row))
                    pid_fields = [name for name in schema if 'Process Name ( PID)' in name]
                    if not any(target_pid(event.get(name, '')) == pid for name in pid_fields):
                        continue
                    # Context-switch counterpart names are unrelated to this experiment.
                    for name in pid_fields:
                        if target_pid(event.get(name, '')) != pid:
                            row[schema.index(name)] = 'other (0)'
                    writer.writerow(row)
                    counts[kind] = counts.get(kind, 0) + 1
            process.stdout.close()
            if process.wait(timeout=60):
                raise RuntimeError('xperf dumper failed; inspect dumper-errors.txt')
    with etl.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'source_etl': str(etl), 'source_etl_sha256': digest,
                'source_etl_bytes': etl.stat().st_size, 'trace_start_unix_s': began,
                'request_start_trace_us': start_us, 'request_end_trace_us': end_us,
                'event_counts': counts, 'sample_windows_us': windows,
                'window_note': 'Four fixed 1-second windows; not a full-request time breakdown.',
                'alignment_note': 'ETL UTC header to request wall clock; validate against recorded xperf marker command bounds.'}
    (folder / 'export-info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    export(parser.parse_args().folder)
