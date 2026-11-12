"""Judge actual complete generations with official EvalPlus in the local VM."""
import argparse
import base64
import gzip
import fcntl
import hashlib
import json
import time
from pathlib import Path
from tokenizers import Tokenizer

from eval_code_tasks import datasets, stopping, completion
from evaluate import digest
from evaluate_suite import now, save_meta
from sandbox import Sandbox


def execute(sandbox, dataset, problems, samples, sanitize=True):
    source_lock = json.loads(Path('evalplus_source.lock.json').read_text())
    payload = dict(dataset=dataset, problems=problems, samples=samples, sanitize=sanitize,
                   package_sources=source_lock['package_sources'])
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    code = 'PAYLOAD_BASE64 = '+repr(packed)+'\n'+Path('evalplus_driver.py').read_text()
    execution = sandbox.run(code, seconds=120, memory_mb=2048, cpu_seconds=90, output_bytes=1_048_576)
    if execution['status'] != 'exited_zero':
        raise RuntimeError(f"Official harness infrastructure failure: {execution['job_directory']}: {execution['status']}")
    markers = [line.removeprefix('QWEN_EVALPLUS_RESULT ') for line in execution['output'].splitlines() if line.startswith('QWEN_EVALPLUS_RESULT ')]
    if len(markers) != 1:
        raise ValueError('Missing/ambiguous official judge result; retain execution log')
    result = json.loads(markers[0])
    if result['dataset'] != dataset or result['package_sources'] != source_lock['package_sources'] or len(result['results']) != len(samples):
        raise ValueError('Official judge result provenance mismatch')
    for row, sample in zip(result['results'], samples, strict=True):
        if row['task_id'] != sample['task_id'] or row['raw_solution_sha256'] != hashlib.sha256(sample['solution'].encode()).hexdigest():
            raise ValueError('Judge changed sample identity')
    return result, execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions');p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=int, default=1200);p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 1 <= a.seconds <= 1200:p.error('seconds must be 1..1200')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = json.loads(Path(a.predictions+'.meta.json').read_text())
        if metadata['status'] != 'complete' or not metadata['full_dataset_generated'] or metadata['identity']['suite'] != 'code':
            raise ValueError('Requires complete code generations, not a pilot or capacity failures')
        data = {name: {r['task_id']: r for r in rows} for name, rows in datasets()}
        tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
        if metadata['identity']['tokenizer_sha256'] != digest('.cache/references/tokenizer.json'):
            raise ValueError('Generation tokenizer drift')
        jobs = [];seen = set()
        with Path(a.predictions).open() as prediction_stream:
            for line in prediction_stream:
                row = json.loads(line);dataset = row['task'].removeprefix('evalplus-');task = row['gold']
                problem = data[dataset][task]
                if row['correct'] is not None or row['prompt'] != problem['prompt'] or (dataset, task) in seen:
                    raise ValueError('Unscored unique original code prompts required')
                if not row['output_ids'] or len(row['output_ids']) > metadata['identity']['tokens']:
                    raise ValueError('Actual generated code tokens required')
                if row['input_ids'] != tokenizer.encode(problem['prompt'], add_special_tokens=False).ids:
                    raise ValueError('Code generation input IDs changed')
                raw = tokenizer.decode(row['output_ids'], skip_special_tokens=False)
                if row['raw_prediction'] != raw or row['prediction'] != completion(tokenizer.decode(row['output_ids'], skip_special_tokens=True), stopping(row['task'])):
                    raise ValueError('Code completion differs from actual generated IDs or official stop rules')
                seen.add((dataset, task));jobs.append((dataset, task, problem['prompt']+row['prediction']))
        expected = {(d, task) for d, tasks in data.items() for task in tasks}
        if seen != expected:raise ValueError('Must generate all 164 HumanEval and 378 MBPP tasks before judging this full suite')
        jobs.sort(key=lambda j: (j[0], j[1]))
        sandbox = Sandbox('evalplus',large=True)
        identity = dict(predictions_sha256=digest(a.predictions), generation_identity=metadata['identity'],
            image_id=sandbox.image, source_hashes={f: digest(f) for f in ('judge_evalplus.py', 'evalplus_driver.py', 'evalplus_source.lock.json', 'sandbox.py', 'vm.sh', 'eval_code_tasks.py')})
        out = Path(a.out);meta_path = Path(a.out+'.meta.json');completed = 0
        if a.resume:
            report = json.loads(meta_path.read_text())
            if report['identity'] != identity:raise ValueError('Judge resume identity mismatch')
            with out.open() as stream:
                for line in stream:
                    r = json.loads(line);dataset, task, solution = jobs[completed]
                    if (r['dataset'], r['task_id'], r['raw_solution_sha256']) != (dataset, task, hashlib.sha256(solution.encode()).hexdigest()):
                        raise ValueError('Judge result prefix mismatch')
                    completed += 1
        else:
            if out.exists() or meta_path.exists():raise FileExistsError('Use a new output or explicit resume')
            out.parent.mkdir(parents=True, exist_ok=True);out.touch(exist_ok=False)
            report = dict(identity=identity, started=now(), events=[], note='Official EvalPlus 0.3.1 judge/sanitizer, base and plus tests; one greedy sample/task; private Qwen evaluator unknown')
        report['events'].append(dict(start=now(), starting_items=completed, budget_seconds=a.seconds))
        report.update(status='running', completed_items=completed, target_items=len(jobs));save_meta(meta_path, report)
        tick = time.monotonic()
        with out.open('a') as stream:
            while completed < len(jobs):
                dataset = jobs[completed][0];group = []
                for job in jobs[completed:completed+1]:
                    if job[0] != dataset:break
                    group.append(job)
                samples = [dict(task_id=task, solution=solution) for _, task, solution in group]
                try:
                    result, execution = execute(sandbox, dataset, [data[dataset][task] for _, task, _ in group], samples)
                except Exception as error:
                    report.update(status='infrastructure_failure', completed_items=completed,
                                  failed_tasks=[task for _, task, _ in group], error=str(error))
                    save_meta(meta_path, report)
                    raise
                for item in result['results']:
                    item.update(dataset=dataset, base_pass=item['base']['passed'], plus_pass=item['base']['passed'] and item['plus']['passed'],
                                execution_directory=execution['job_directory'], image_id=execution['image_id'], driver_sha256=execution['code_sha256'])
                    stream.write(json.dumps(item)+'\n');stream.flush();completed += 1
                report.update(completed_items=completed);save_meta(meta_path, report);print(now(), completed, len(jobs), flush=True)
                if time.monotonic()-tick >= a.seconds:break
        scores = {}
        with out.open() as stream:
            for line in stream:
                row = json.loads(line);count = scores.setdefault(row['dataset'], dict(n=0, base_correct=0, plus_correct=0))
                count['n'] += 1;count['base_correct'] += int(row['base_pass']);count['plus_correct'] += int(row['plus_pass'])
        report.update(status='complete' if completed == len(jobs) else 'time_budget', completed_items=completed, tasks=scores)
        if completed == len(jobs):report['four_suite_mean'] = sum(c[k]/c['n'] for c in scores.values() for k in ('base_correct', 'plus_correct'))/4
        report['events'][-1].update(end=now(), ending_items=completed);save_meta(meta_path, report)


if __name__ == '__main__':
    main()
