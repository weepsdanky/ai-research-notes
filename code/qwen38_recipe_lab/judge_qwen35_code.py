"""Run or audit formal Base generations with the pinned local code judges."""
import argparse
import ast
import base64
from datetime import datetime, timedelta
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from qwen35_code_jobs import validated_jobs
from qwen35_quality_data import digest, load, write_json


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def driver_payload(directory, name):
    assignment, original = (directory / 'task.py').read_text().split('\n', 1)
    assert original == Path(name).read_text()
    node, = ast.parse(assignment).body
    assert isinstance(node, ast.Assign)
    target, = node.targets
    assert isinstance(target, ast.Name) and target.id == 'PAYLOAD_BASE64'
    return json.loads(gzip.decompress(base64.b64decode(ast.literal_eval(node.value))))


def verify_evalplus(row, job, image):
    directory = Path(row['execution_directory']).resolve()
    if directory.parent != Path('runs/code-jobs').resolve():
        raise ValueError('Execution directory is outside the local job root')
    execution = load(directory / 'execution.json')
    package = load('evalplus_source.lock.json')['package_sources']
    assert execution['status'] == 'exited_zero' and execution['exit_code'] == 0
    assert execution['cleanup_ok'] and not execution['oom_killed']
    assert row['image_id'] == execution['image_id'] == image
    assert execution['limits'] == dict(wall_seconds=120, memory_mb=2048, cpu_seconds=90,
                                       pids=32, cpus=1, output_bytes=1048576, file_bytes=1048576)
    assert digest(directory / 'task.py') == execution['code_sha256'] == row['driver_sha256']
    payload = driver_payload(directory, 'evalplus_driver.py')
    assert payload == dict(dataset=job['dataset'], problems=[job['problem']],
                           samples=[dict(task_id=job['name'], solution=job['solution'])],
                           sanitize=True, package_sources=package)
    marker = 'QWEN_EVALPLUS_RESULT '
    outputs = [json.loads(line[len(marker):]) for line in execution['output'].splitlines()
               if line.startswith(marker)]
    official, = outputs
    result, = official['results']
    assert official['dataset'] == job['dataset'] and official['package_sources'] == package
    assert official['fast_check'] is True and result['task_id'] == job['name']
    assert all(row[key] == value for key, value in result.items())
    assert row['raw_solution_sha256'] == sha(job['solution'])
    assert row['solution_sha256'] == sha(row['sanitized_solution'])
    assert row['base_pass'] == row['base']['passed']
    assert row['plus_pass'] == (row['base']['passed'] and row['plus']['passed'])
    assert all(row[mode]['total_inputs'] == len(job['problem'][mode+'_input'])
               for mode in ('base', 'plus'))


def check_row(row, job, image):
    if job['suite'] == 'code':
        assert row['dataset'] == job['dataset'] and row['task_id'] == job['name']
        verify_evalplus(row, job, image)
    else:
        from audit_multipl_results import verify_execution
        from multipl_judge import MODULES

        assert row['language'] == job['dataset'] and row['name'] == job['name']
        assert row['completion_sha256'] == sha(job['completion'])
        verify_execution(row, job['problem'], job['completion'],
                         load('multipl_judge_sources.lock.json'), Path('runs/code-jobs').resolve(), image)
        payload = driver_payload(Path(row['execution_directory']), 'multipl_driver.py')
        assert payload['language'] == job['dataset'] and payload['module'] == MODULES[job['dataset']]
        assert row['tests_completed'] == (payload['sentinel'] in row['stdout'].splitlines())
        assert row['passed'] == (row['status'] == 'OK' and row['exit_code'] == 0 and row['tests_completed'])


def execute_job(sandbox, job):
    if job['suite'] == 'code':
        from judge_evalplus import execute

        result, execution = execute(sandbox, job['dataset'], [job['problem']],
                                    [dict(task_id=job['name'], solution=job['solution'])])
        row, = result['results']
        return dict(row, dataset=job['dataset'], base_pass=row['base']['passed'],
                    plus_pass=row['base']['passed'] and row['plus']['passed'],
                    execution_directory=execution['job_directory'],
                    image_id=execution['image_id'], driver_sha256=execution['code_sha256'])

    from multipl_judge import execute

    row = execute(sandbox, job['dataset'], job['problem'], job['completion'])
    return dict(row, language=job['dataset'], name=job['name'],
                completion_sha256=sha(job['completion']))


def checked_prefix(out, jobs, image):
    totals = {}
    count = 0
    with out.open() as stream:
        for line in stream:
            if count == len(jobs) or not line.endswith('\n'):
                raise ValueError('Unexpected or incomplete result row')
            row = json.loads(line)
            job = jobs[count]
            check_row(row, job, image)
            score = totals.setdefault(job['dataset'], {'items': 0})
            score['items'] += 1
            keys = ('base_pass', 'plus_pass') if job['suite'] == 'code' else ('passed',)
            for key in keys:
                score[key] = score.get(key, 0) + int(row[key])
            count += 1
    return count, totals


def resources_ready():
    pressure = int(subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))
    memory = subprocess.check_output(['memory_pressure', '-Q'], text=True)
    return pressure == 1 and int(re.search(r'free percentage:\s*(\d+)%', memory)[1]) >= 55


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions')
    parser.add_argument('generation_audit')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--audit-out', type=Path, help='Recheck saved execution evidence without starting a VM')
    args = parser.parse_args()
    if not args.audit_out and (args.deadline is None or args.deadline.tzinfo is None or
                               not 150 < args.seconds <= 1180):
        parser.error('Judging requires an aware --deadline and --seconds in 151..1180')
    if args.out.parent.resolve() != Path('runs').resolve():
        parser.error('Store model outputs in the local runs directory')
    meta = Path(str(args.out)+'.meta.json')
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance, jobs = validated_jobs(args.predictions, args.generation_audit)
        suite = jobs[0]['suite']
        image_name = 'evalplus' if suite == 'code' else 'multipl'
        image = load('code_images.lock.json')[image_name]['image_id']
        identity = dict(generation_provenance=provenance, judge_sha256=digest(__file__), image_id=image,
                        helper_sha256=digest('audit_multipl_results.py') if suite == 'multipl' else None)
        if args.resume or args.audit_out:
            report = load(meta)
            if report['identity'] != identity:
                raise ValueError('Resume/audit identity mismatch')
            count, scores = checked_prefix(args.out, jobs, image)
        else:
            if args.out.exists() or meta.exists():
                raise FileExistsError('Select a new output or use --resume')
            report = dict(identity=identity, target_items=len(jobs), events=[])
            count, scores = 0, {}

        if args.audit_out:
            result = dict(identity=identity, status='passed_execution_evidence_audit',
                          judged_sha256=digest(args.out), meta_sha256=digest(meta),
                          items=count, full_judged=count == len(jobs),
                          tasks=scores, raw_only=suite == 'multipl')
            if suite == 'code' and result['full_judged']:
                result['four_suite_mean'] = sum(row[key]/row['items'] for row in scores.values()
                                                for key in ('base_pass', 'plus_pass'))/4
            with args.audit_out.open('x') as stream:
                json.dump(result, stream, indent=2)
            return

        if count == len(jobs):
            report.update(status='complete', completed_items=count)
            write_json(meta, report)
            return

        now = datetime.now().astimezone()
        if args.deadline > now + timedelta(hours=2):
            parser.error('Use the current execution window')
        end = min(args.deadline - timedelta(seconds=600), now + timedelta(seconds=args.seconds))
        if not resources_ready() or (end - now).total_seconds() <= 150:
            raise RuntimeError('Insufficient resources or time for a bounded local job')
        from sandbox import Sandbox

        sandbox = Sandbox(image_name, large=suite == 'code')
        assert sandbox.image == image
        if not args.resume:
            args.out.touch(exist_ok=False)
        event = dict(started=now.isoformat(), starting_items=count, deadline=args.deadline.isoformat())
        report['events'].append(event)
        report.update(status='running', completed_items=count, raw_only=suite == 'multipl')
        write_json(meta, report)
        try:
            with args.out.open('a') as stream:
                for job in jobs[count:]:
                    if (end - datetime.now().astimezone()).total_seconds() <= 150:
                        report['status'] = 'time_budget'
                        break
                    if not resources_ready():
                        report['status'] = 'memory_pressure'
                        break
                    row = execute_job(sandbox, job)
                    check_row(row, job, image)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    stream.flush()
                    count += 1
                    report['completed_items'] = count
                    write_json(meta, report)
                else:
                    report['status'] = 'complete'
        except BaseException:
            report['status'] = 'execution_interrupted'
            raise
        finally:
            event.update(ended=datetime.now().astimezone().isoformat(), ending_items=count,
                         status=report['status'])
            write_json(meta, report)


if __name__ == '__main__':
    main()
