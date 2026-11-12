"""Resumable full/pilot generation evaluation with explicit data coverage.

Public shot counts are preserved. Prompts, extraction, decode/context budgets
are declared Mac choices; this is not the unpublished official Qwen harness.
"""
import argparse,fcntl,fnmatch,hashlib,json,time,importlib,platform
from datetime import datetime
from pathlib import Path
import torch
from tokenizers import Tokenizer
from eval_tasks import load_tasks,correct
from evaluate import digest,next_logits
from inference import Inference
from model import Model,Config
from train import device_for


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


def save_meta(path,metadata):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--tasks',nargs='+',required=True,help='Task names or quoted glob patterns')
    p.add_argument('--suite',choices=['core','extra'],default='core',help='Extra adapters are supplied separately')
    scope=p.add_mutually_exclusive_group();scope.add_argument('--full',action='store_true');scope.add_argument('--limit',type=int)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--context',type=int,default=2048)
    p.add_argument('--seconds',type=int,default=1200);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--backend',choices=['incremental','reference'],default='incremental')
    p.add_argument('--prefill-chunk',type=int,default=0,help='Bound prefill query working set; zero uses full prefill')
    p.add_argument('--retry-capacity-from',help='Retry only capacity failures in this completed prediction file')
    a=p.parse_args()
    if min(a.tokens,a.context,a.seconds,a.limit if a.limit is not None else 2)<1 or a.seconds>1200:p.error('Positive budgets required, seconds <=1200')
    if not 0<=a.prefill_chunk<=512 or (a.prefill_chunk and a.backend!='incremental'):p.error('Chunk size 0..512 requires incremental backend')
    if a.retry_capacity_from and not a.full:p.error('Capacity retry requires --full; it remains a targeted subset')
    torch.set_num_threads(1);torch.manual_seed(42)
    out=Path(a.out);meta_path=Path(a.out+'.meta.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    lock=Path(a.out+'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another evaluator owns this output file')
    adapter=importlib.import_module('eval_tasks' if a.suite=='core' else 'eval_extra_tasks')
    score=adapter.correct
    tasks=[t for t in adapter.load_tasks() if any(fnmatch.fnmatchcase(t.name,pattern) for pattern in a.tasks)]
    if not tasks:p.error('No tasks matched')
    if any(not any(fnmatch.fnmatchcase(t.name,pattern) for t in tasks) for pattern in a.tasks):p.error('An unmatched task pattern was supplied')
    limit=None if a.full else (a.limit or 2)
    jobs=[(task,index,prompt,gold) for task in tasks for index,(prompt,gold) in enumerate(task.rows[:limit])]
    retry_hash=None
    if a.retry_capacity_from:
        source_meta=json.loads(Path(a.retry_capacity_from+'.meta.json').read_text())
        if source_meta['status']!='complete':raise ValueError('Retry source must be terminal and complete')
        if source_meta['identity']['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Retry checkpoint differs')
        retry_hash=digest(a.retry_capacity_from);failed={}
        with Path(a.retry_capacity_from).open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['stop']=='context_capacity':failed[(r['task'],r['index'])]=(r['prompt_sha256'],r['gold'])
        jobs=[j for j in jobs if (j[0].name,j[1]) in failed]
        if not jobs:p.error('No matching capacity failures')
        for task,index,prompt,gold in jobs:
            if failed[(task.name,index)]!=(hashlib.sha256(prompt.encode()).hexdigest(),gold):raise ValueError('Retry inputs changed')
        tasks=[t for t in tasks if any(j[0].name==t.name for j in jobs)]
    content=hashlib.sha256()
    for task,index,prompt,gold in jobs:
        content.update(json.dumps((task.name,index,task.shots,task.cot,task.kind,prompt,gold),ensure_ascii=False).encode())
    core={name:digest(name) for name in ('model.py','optim.py','polar_coefficients.json','train.py')}
    identity=dict(checkpoint_sha256=digest(a.checkpoint),content_sha256=content.hexdigest(),
        evaluator_sha256=digest(__file__),adapters_sha256=digest(adapter.__file__),common_scoring_sha256=digest('eval_tasks.py'),inference_sha256=digest('inference.py'),
        runtime_model_sources=core,device=a.device,backend=a.backend,tokens=a.tokens,context=a.context,
        prefill_chunk=a.prefill_chunk,retry_source_sha256=retry_hash,suite=a.suite,
        tasks=[t.name for t in tasks],scope='targeted capacity retry' if retry_hash else ('full selected datasets' if a.full else f'pilot limit {limit} per task'))
    counts={t.name:dict(n=0,correct=0,capacity_failures=0,total_dataset=len(t.rows),target=sum(j[0].name==t.name for j in jobs),shots=t.shots,cot=t.cot,kind=t.kind) for t in tasks}
    completed=0
    if a.resume:
        metadata=json.loads(meta_path.read_text())
        if metadata['identity']!=identity:raise ValueError('Resume identity mismatch: retain the exact evaluator, weights, input set and protocol')
        with out.open() as stream:
            for line in stream:
                record=json.loads(line)
                if completed>=len(jobs):raise ValueError('Unexpected extra prediction')
                task,index,prompt,gold=jobs[completed]
                if (record['task'],record['index'],record['prompt_sha256'],record['gold'])!=(task.name,index,hashlib.sha256(prompt.encode()).hexdigest(),gold):
                    raise ValueError('Prediction/input prefix mismatch')
                if record['correct']!=score(record['prediction'],gold,task.kind):raise ValueError('Stored score mismatch')
                counts[task.name]['n']+=1;counts[task.name]['correct']+=int(record['correct'])
                counts[task.name]['capacity_failures']+=int(record['stop']=='context_capacity');completed+=1
    else:
        if out.exists() or meta_path.exists():raise FileExistsError('Use a new output path or explicit --resume')
        out.touch(exist_ok=False)
        metadata=dict(identity=identity,started=now(),events=[],source_lock_sha256_at_start=digest('eval_sources.lock.json'),
            environment=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform()),
            tokenizer_sha256=digest('.cache/references/tokenizer.json'),
            protocol='greedy; no truncation of few-shot; failures retained; limits explicit; not official prompts/decoding',
            aggregation='per-task accuracy, unweighted subtask macro and item micro; BBH family aggregation not inferred')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if saved['step']!=saved['args']['steps']:raise ValueError('Use a completed phase checkpoint')
    if saved['source_hashes']!=core:raise ValueError('Runtime training core differs from checkpoint source')
    saved.pop('optimizers',None)
    device=device_for(a.device);model=Model(Config(**saved['config'])).to(device).eval()
    model.load_state_dict(saved.pop('model'))
    stage='sparse' if saved['args']['stage']=='sparse' else 'pretrain'
    metadata.update(checkpoint_step=saved['step'],config=saved['config'],training_context=saved['args']['seq'])
    del saved
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json')
    runner=Inference(model,stage,prefill_chunk=a.prefill_chunk)
    def logits(ids,cache):
        return runner(ids,cache)[0] if a.backend=='incremental' else next_logits(model,ids,stage,cache)
    metadata['events'].append(dict(start=now(),resume=a.resume,starting_items=completed,budget_seconds=a.seconds))
    metadata.update(status='running',completed_items=completed,tasks=counts);save_meta(meta_path,metadata)
    started=time.monotonic()
    with out.open('a') as stream:
        for task,index,prompt,gold in jobs[completed:]:
            ids=tokenizer.encode(prompt,add_special_tokens=False).ids;generated=[];tick=time.perf_counter()
            record=dict(task=task.name,index=index,shots=task.shots,cot=task.cot,answer_kind=task.kind,
                prompt=prompt,input_ids=ids,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),gold=gold,
                prompt_tokens=len(ids),context_extrapolation=len(ids)>metadata['training_context'])
            stop='max_new_tokens'
            if len(ids)+a.tokens>a.context:stop='context_capacity'
            else:
                cache={};last=logits(torch.tensor([ids],device=device),cache)
                record['prefill_seconds']=time.perf_counter()-tick
                decode=time.perf_counter()
                for step in range(a.tokens):
                    token=int(last.argmax().item());generated.append(token)
                    if token==model.config.eos:stop='eos';break
                    if step+1<a.tokens:last=logits(torch.tensor([[token]],device=device),cache)
                record['decode_seconds']=time.perf_counter()-decode
            text=tokenizer.decode(generated,skip_special_tokens=False)
            record.update(output_ids=generated,prediction=text,correct=score(text,gold,task.kind),stop=stop,
                unused_token_ids=[t for t in generated if tokenizer.id_to_token(t) is None],seconds=time.perf_counter()-tick)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            count=counts[task.name];count['n']+=1;count['correct']+=int(record['correct']);count['capacity_failures']+=int(stop=='context_capacity')
            completed+=1
            metadata.update(completed_items=completed,tasks=counts)
            if completed%25==0:save_meta(meta_path,metadata);print(now(),completed,len(jobs),task.name,flush=True)
            if time.monotonic()-started>=a.seconds:break
    for count in counts.values():count['accuracy']=count['correct']/count['n'] if count['n'] else None
    metadata.update(status='complete' if completed==len(jobs) else 'time_budget',completed_items=completed,tasks=counts,
        full_dataset_attempted=completed==len(jobs) and a.full and not retry_hash,
        full_dataset_generated=completed==len(jobs) and a.full and not retry_hash and not any(c['capacity_failures'] for c in counts.values()))
    active=[c for c in counts.values() if c['n']]
    metadata['macro_accuracy']=sum(c['accuracy'] for c in active)/len(active) if active else None
    metadata['micro_accuracy']=sum(c['correct'] for c in active)/sum(c['n'] for c in active) if active else None
    metadata['events'][-1].update(end=now(),ending_items=completed)
    save_meta(meta_path,metadata);print(metadata['status'],completed,len(jobs),flush=True)


if __name__=='__main__':main()
