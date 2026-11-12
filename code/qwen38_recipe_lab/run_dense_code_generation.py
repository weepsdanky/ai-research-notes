"""Generate and independently audit the frozen Dense code benchmark inputs."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from qwen35_quality_data import adapter_for, decoded, digest, load, prepare_jobs, write_json
from run_control_quality import LOCK, memory, normal, review
from run_dense_text_guarded import native, pressure

PROFILE = Path('profiles/paper_dense_bounded_evaluation.json')
BINDING = Path('runs/paper-dense-evaluation-binding.json')
REPAIRED = Path('profiles/paper_dense_multipl_repaired_evaluation.json')


def repaired_job():
    job = load(REPAIRED)
    assert digest(job['primary_audit']) == job['primary_audit_sha256']
    assert digest('multipl_repairs.lock.json') == job['repair_lock_sha256']
    audit = load(job['primary_audit'])
    assert audit['items'] == job['target_items'] == 3 and audit['not_official_protocol']
    assert audit['all_prompts_ids_raw_decoding_and_stopped_completions_verified']
    assert audit['file_sha256'] == digest(job['primary_meta'].removesuffix('.meta.json'))
    return job


def preflight(name):
    from tokenizers import Tokenizer

    profile, binding = load(PROFILE), load(BINDING)
    if name == 'multipl-repaired':
        job = repaired_job()
    else:
        job, = [row for row in profile['jobs'] if row['name']==name]
    assert job['suite'] in ('code','multipl','multipl_repaired')
    assert digest('profiles/paper_dense_evaluation.json')==profile['parent_profile_sha256']==binding['profile_sha256']
    assert digest('evaluate_dense_bounded.py')==profile['engine_sha256']
    assert digest('bounded_dense_inference.py')==profile['kernel_sha256']
    checkpoint = profile['checkpoint']
    assert checkpoint==binding['checkpoint'] and digest(checkpoint)==binding['checkpoint_sha256']
    assert digest(binding['checkpoint_audit'])==binding['checkpoint_audit_sha256']
    training = load(binding['checkpoint_audit'])
    assert training['phase_complete'] and training['step']==8000
    assert all(digest(path)==sha for path,sha in training['source_hashes'].items())
    assert digest(job['primary_meta'])==job['primary_meta_sha256']
    primary = load(job['primary_meta'])
    assert primary['full_dataset_generated'] and primary['completed_items']==job['target_items']
    tokenizer_path = '.cache/references/tokenizer.json'
    tokenizer_sha = digest(tokenizer_path)
    assert tokenizer_sha==primary['identity']['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(tokenizer_path)
    adapter, tasks = adapter_for(job)
    rows, content, _ = prepare_jobs(tasks,tokenizer,job)
    assert content==job['primary_content_sha256']==primary['identity']['content_sha256']
    assert all(job[key]==primary['identity'][key] for key in ('tokens','context','suite'))
    stops = {task.name:adapter.stopping(task.name) for task in tasks}
    assert stops==primary['identity']['stop_strings']
    evidence = dict(profile_sha256=digest(PROFILE),binding_sha256=digest(BINDING),checkpoint=checkpoint,
                    checkpoint_sha256=binding['checkpoint_sha256'],training_audit_sha256=binding['checkpoint_audit_sha256'],
                    primary_meta_sha256=job['primary_meta_sha256'],tokenizer_sha256=tokenizer_sha,
                    content_sha256=content,items=len(rows),max_prompt_tokens=max(len(row[4]) for row in rows),
                    engine_sha256=profile['engine_sha256'],kernel_sha256=profile['kernel_sha256'])
    if name == 'multipl-repaired':
        evidence['repaired_profile_sha256'] = digest(REPAIRED)
    return evidence,job,rows,adapter,tokenizer,stops


def audit_rows(path, job, rows, adapter, tokenizer, stops):
    size = 0
    with path.open() as stream:
        for size,line in enumerate(stream,1):
            task,index,prompt,gold,ids = rows[size-1]
            row = json.loads(line)
            assert (row['task'],row['index'],row['prompt'],row['gold'],row['input_ids'])==(task.name,index,prompt,gold,ids)
            assert row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            tokens = row['output_ids']
            assert 0<len(tokens)<=job['tokens'] and all(type(x) is int and 0<=x<248320 for x in tokens)
            raw,prediction = decoded(adapter,tokenizer,tokens,stops[task.name])
            assert row['raw_prediction']==raw and row['prediction']==prediction
            assert row['correct'] is None and adapter.correct(prediction,gold,task.kind) is None
            reason = row['stop']
            if reason=='eos':
                assert tokens[-1]==248044 and 248044 not in tokens[:-1]
            elif reason=='stop_string':
                assert 248044 not in tokens
                assert any(stop in tokenizer.decode(tokens[-64:],skip_special_tokens=False) for stop in stops[task.name])
            else:
                assert reason=='max_new_tokens' and len(tokens)==job['tokens'] and 248044 not in tokens
    return size


def audit(saved, job, rows, adapter, tokenizer, stops, evidence):
    identity = saved['identity']
    assert all(identity[key]==job[key] for key in ('tokens','context','suite','batch_size','prefill_chunk',
                                                     'grouped_gqa','gdn_scan'))
    assert bool(identity['prefix_cache'])==job['prefix_cache']
    assert identity['scope']=='full selected datasets' and identity['device']=='cpu' and identity['stop_pressure']==2
    assert identity['stop_strings']==stops
    assert all(identity[key]==evidence[key] for key in ('checkpoint_sha256','content_sha256','tokenizer_sha256'))
    assert identity['dense_execution_variant']['source_sha256']==evidence['kernel_sha256']
    for key in ('source_hashes','runtime_model_sources','chunk_sources'):
        assert all(digest(path)==sha for path,sha in identity[key].items())
    if identity['prefix_cache']:
        assert digest('prefix_inference.py')==identity['prefix_cache']['source_sha256']
    size = audit_rows(Path(job['output']),job,rows,adapter,tokenizer,stops)
    assert size==saved['completed_items']
    complete = size==len(rows)
    assert complete==saved['full_dataset_generated']
    assert saved['scoring_status']=='pending official code harness'
    return dict(items=size,target=len(rows),complete=complete,judged=False,
                predictions_sha256=digest(job['output']),meta_sha256=digest(job['output']+'.meta.json'),
                native_status=saved['status'])


def run(args, evidence, job, rows, adapter, tokenizer, stops):
    out = Path(job['output'])
    meta = Path(str(out)+'.meta.json')
    assert out.exists()==meta.exists()==bool(args.prior)
    status = dict(status='admitting', pid=os.getpid(), job=args.job, evidence=evidence,
                  deadline=args.deadline.isoformat(), reviews=[])
    write_json(args.receipt,status)
    old = child = None
    try:
        if args.prior:
            prior = load(args.prior)
            assert prior['status']=='saved' and prior['job']==args.job and prior['evidence']==evidence
            old_deadline = datetime.fromisoformat(prior['deadline'])
            assert args.deadline<=old_deadline or datetime.now().astimezone()>=old_deadline
            old = prior['generation']
            assert not old['complete'] and old['resume_reason'] in ('time_budget','memory_pressure')
            assert digest(out)==old['predictions_sha256'] and digest(meta)==old['meta_sha256']
            before = out.stat().st_size
        if (old and old['resume_reason']=='memory_pressure') or not normal(memory()):
            bound = [Path(evidence['checkpoint']),PROFILE]
            if args.job == 'multipl-repaired':
                bound.extend([REPAIRED, Path('multipl_repairs.lock.json'), Path(job['primary_audit'])])
            if old:
                bound.extend([out,meta,args.prior])
            if not review(status,args.receipt,args.deadline,bound):
                status['status']='resource_review_incomplete'
                return
        with LOCK.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert normal(memory())
            seconds = min(args.seconds,int((args.deadline-datetime.now().astimezone()).total_seconds())-600)
            assert seconds>=30
            command = [sys.executable,'evaluate_dense_bounded.py',evidence['checkpoint'],'--out',str(out),
                       '--suite',job['suite'],'--tasks',*job['tasks'],'--full','--tokens',str(job['tokens']),
                       '--context',str(job['context']),'--batch-size',str(job['batch_size']),
                       '--prefill-chunk',str(job['prefill_chunk']),'--gdn-scan',job['gdn_scan'],
                       '--device','cpu','--seconds',str(seconds),'--stop-pressure','2']
            if job['prefix_cache']:
                command.append('--prefix-cache')
            if job['grouped_gqa']:
                command.append('--grouped-gqa')
            if old:
                assert digest(out)==old['predictions_sha256'] and digest(meta)==old['meta_sha256']
                command.append('--resume')
            log = args.receipt.with_suffix('.log')
            with log.open('x') as stream:
                child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                status.update(status='running',child_pid=child.pid,command=command,log=str(log))
                write_json(args.receipt,status)
                cutoff = args.deadline-timedelta(seconds=600)
                while child.poll() is None:
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        level = pressure()
                        if level>=2 or datetime.now().astimezone()>=cutoff:
                            reason = 'memory_pressure' if level>=2 else 'time_budget'
                            status['stop_request'] = dict(reason=reason,pressure=level)
                            child.send_signal(signal.SIGINT)
                            try:
                                child.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                status['returncode'] = child.returncode
            assert status['returncode']==0
            if old:
                with out.open('rb') as stream:
                    h=hashlib.sha256()
                    remaining=before
                    while remaining:
                        chunk=stream.read(min(remaining,1<<20))
                        assert chunk
                        h.update(chunk)
                        remaining-=len(chunk)
                assert h.hexdigest()==old['predictions_sha256']
            result = audit(load(meta),job,rows,adapter,tokenizer,stops,evidence)
            reason = result['native_status']
            if reason=='interrupted_signal':
                reason = status['stop_request']['reason']
            result['resume_reason'] = reason
            assert result['complete'] or reason in ('time_budget','memory_pressure')
            assert not old or result['items']>old['items']
            status.update(status='saved',generation=result,swap=native(['sysctl','vm.swapusage']).strip())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status.update(status='needs_diagnosis',error=repr(exc))
        raise
    finally:
        status['ended']=datetime.now().astimezone().isoformat()
        write_json(args.receipt,status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=('code','multipl','multipl-repaired'))
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    args = parser.parse_args()
    assert not args.receipt.exists() and args.receipt.parent.resolve()==Path('runs').resolve()
    evidence,job,rows,adapter,tokenizer,stops = preflight(args.job)
    if args.prepare:
        assert not args.prior
        write_json(args.receipt,dict(status='verified_public_prompts',evidence=evidence,model_run=False))
        return
    now = datetime.now().astimezone()
    assert args.deadline and now+timedelta(seconds=660)<args.deadline<=now+timedelta(hours=2)
    assert 30<=args.seconds<=1100
    assert not any(k in os.environ for k in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'))
    run(args,evidence,job,rows,adapter,tokenizer,stops)


if __name__=='__main__':
    main()
