"""Score all eight complete MultiPL-E generations with locked public runners."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
from tokenizers import Tokenizer
from eval_multipl_tasks import datasets, completion, stopping
from evaluate import digest
from evaluate_suite import now, save_meta
from multipl_judge import execute
from sandbox import Sandbox


def validated_jobs(predictions):
    metadata = json.loads(Path(predictions+'.meta.json').read_text())
    if metadata['status']!='complete' or not metadata['full_dataset_generated'] or metadata['identity']['suite']!='multipl':raise ValueError('Requires complete eight-language generation')
    tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
    if digest('.cache/references/tokenizer.json')!=metadata['identity']['tokenizer_sha256']:raise ValueError('Tokenizer drift')
    data = {(lang,r['name']):r for lang,rows in datasets().items() for r in rows}
    jobs = {};rows = [json.loads(line) for line in Path(predictions).read_text().splitlines()]
    for row in rows:
        key = row['task'].removeprefix('multipl-'),row['gold'];problem = data[key]
        if key in jobs or row['correct'] is not None or row['prompt']!=problem['prompt']:raise ValueError('Prediction identity mismatch')
        ids = row['output_ids']
        if not 0 < len(ids) <= metadata['identity']['tokens'] or row['input_ids']!=tokenizer.encode(problem['prompt'],add_special_tokens=False).ids:raise ValueError('Generated/input IDs mismatch')
        if row['raw_prediction']!=tokenizer.decode(ids,skip_special_tokens=False) or row['prediction']!=completion(tokenizer.decode(ids,skip_special_tokens=True),stopping(row['task'])):raise ValueError('Raw/completion ID drift')
        jobs[key] = row['prediction']
    if jobs.keys()!=data.keys():raise ValueError('All 1277 unique public tasks required')
    jobs = sorted((lang,name,value) for (lang,name),value in jobs.items())
    return metadata,data,jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions');p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--raw-only',action='store_true',help='Record unadjusted harness outcomes; do not report model accuracy before scaffold validation')
    a = p.parse_args()
    if not a.raw_only:p.error('Original locked data has three independently invalid scaffolds; use --raw-only. The separate Mac-repaired variant is scored by score_multipl_repairs.py.')
    if not 1 <= a.seconds <= 1200:p.error('seconds must be 1..1200')
    out = Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with Path(a.out+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metadata,data,jobs = validated_jobs(a.predictions)
        sandbox = Sandbox('multipl')
        identity = dict(predictions_sha256=digest(a.predictions),generation_identity=metadata['identity'],image_id=sandbox.image,raw_only=a.raw_only,
            source_hashes={f:digest(f) for f in ['judge_multipl.py','multipl_judge.py','multipl_driver.py','multipl_judge_sources.lock.json','eval_multipl_tasks.py','multipl_sources.lock.json','sandbox.py','vm.sh']})
        meta = Path(a.out+'.meta.json');done = 0
        if a.resume:
            report = json.loads(meta.read_text())
            if report['identity']!=identity:raise ValueError('Judge resume source identity mismatch')
            for line in out.read_text().splitlines():
                row = json.loads(line);lang,name,value = jobs[done]
                if (row['language'],row['name'],row['completion_sha256'])!=(lang,name,hashlib.sha256(value.encode()).hexdigest()):raise ValueError('Result prefix mismatch')
                done += 1
        else:
            if out.exists() or meta.exists():raise FileExistsError('New output or explicit --resume required')
            out.touch(exist_ok=False)
            report = dict(identity=identity,started=now(),events=[],not_model_accuracy=a.raw_only,protocol='Publisher eval_script and tests; test-completion sentinel safeguard; one greedy sample/task, not the publisher 20-sample temperature-0.2 estimate; private Qwen evaluator unknown')
        report['events'].append(dict(start=now(),starting_items=done,budget_seconds=a.seconds))
        report.update(status='running',completed_items=done,target_items=len(jobs));save_meta(meta,report)
        tick = time.monotonic()
        with out.open('a') as stream:
            for lang,name,value in jobs[done:]:
                try:result = execute(sandbox,lang,data[lang,name],value)
                except Exception as error:
                    report.update(status='infrastructure_failure',completed_items=done,failed_task=[lang,name],error=str(error));save_meta(meta,report);raise
                result.update(language=lang,name=name,completion_sha256=hashlib.sha256(value.encode()).hexdigest())
                stream.write(json.dumps(result)+'\n');stream.flush();done += 1
                report.update(completed_items=done);save_meta(meta,report);print(now(),done,len(jobs),flush=True)
                if time.monotonic()-tick>=a.seconds:break
        scores = {}
        for line in out.read_text().splitlines():
            row = json.loads(line);score = scores.setdefault(row['language'],dict(n=0,correct=0))
            score['n'] += 1;score['correct'] += int(row['passed'])
        report.update(status='complete' if done==len(jobs) else 'time_budget',completed_items=done,tasks=scores)
        if done==len(jobs) and not a.raw_only:
            report['eight_language_macro_pass_fraction'] = sum(s['correct']/s['n'] for s in scores.values())/8
            report['micro_pass_fraction'] = sum(s['correct'] for s in scores.values())/len(jobs)
        report['events'][-1].update(end=now(),ending_items=done);save_meta(meta,report)


if __name__=='__main__':main()
