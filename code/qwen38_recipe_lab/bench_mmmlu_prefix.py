"""Compare cold and subject-prefix inference on completed real Chinese batches.

Preparation is streaming/CPU data work. Neural checks must run separately from
the formal evaluator. This is an equivalence probe, not a full speed benchmark.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DATA=Path('runs/mmmlu-prefix-cases.json')
PREDICTIONS=Path('runs/eval-sparse-mmmlu-zh_cn-full.jsonl')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    meta=json.loads(Path(str(PREDICTIONS)+'.meta.json').read_text());assert meta['full_dataset_generated']
    spec=next(s for s in json.loads(Path('mmmlu_sources.lock.json').read_text()) if s['language']=='ZH-CN')
    source=Path('data/mmmlu')/spec['name'];assert digest(source)==spec['sha256']
    with source.open(encoding='utf-8-sig',newline='') as stream:subjects=[r['Subject'] for r in csv.DictReader(stream)]
    groups=defaultdict(list)
    with PREDICTIONS.open() as stream:
        for line in stream:
            r=json.loads(line);groups[r['batch_start']].append((r['index'],r['prompt_tokens'],r['batch_items'],len(r['output_ids'])))
    homogeneous=defaultdict(list);mixed=[];single=[]
    for start,rows in groups.items():
        tags={subjects[i] for i,_,_,_ in rows}
        if not all(count==64 and size==len(rows) for _,_,size,count in rows):continue
        if len(rows)==8:
            if len(tags)==1:homogeneous[next(iter(tags))].append(start)
            else:mixed.append(start)
        if len(rows)==1 and rows[0][1]>1500:single.append(start)
    repeated=next(tag for tag,starts in homogeneous.items() if len(starts)>=2)
    other=next(tag for tag in homogeneous if tag!=repeated)
    selected=[*homogeneous[repeated][:2],homogeneous[other][0],max(single,key=lambda s:groups[s][0][1])]
    if mixed:selected.append(mixed[0])
    cases={s:[] for s in selected};wanted={subjects[i] for s in selected for i,_,_,_ in groups[s]}
    prefix_rows=defaultdict(list)
    with PREDICTIONS.open() as stream:
        for line in stream:
            r=json.loads(line);subject=subjects[r['index']]
            if subject in wanted:prefix_rows[subject].append(r['input_ids'])
            if r['batch_start'] in cases:cases[r['batch_start']].append({**{k:r[k] for k in ['index','input_ids','output_ids','batch_start']},'subject':subject})
    # Avoid importing torch just to find a prefix of integer lists.
    prefixes={}
    for tag,rows in prefix_rows.items():
        prefix=rows[0].copy()
        for row in rows[1:]:
            n=0
            while n<min(len(prefix),len(row)) and prefix[n]==row[n]:n+=1
            del prefix[n:]
        prefixes[tag]=prefix[:len(prefix)//128*128]
    result=dict(predictions_sha256=digest(PREDICTIONS),checkpoint_sha256=meta['identity']['checkpoint_sha256'],
        cases=list(cases.values()),prefixes=prefixes,selection='Two full batches sharing one subject, another subject, longest single, and a mixed fallback; selected by input shape only')
    DATA.write_text(json.dumps(result)+'\n');print('Prepared',sum(map(len,cases.values())),'real cases')


def check(checkpoint, out, reference):
    import time
    import torch
    from model import Model,Config
    from inference import Inference
    from prefill_inference import PrefillInference
    from prefix_inference import PrefixInference
    data=json.loads(DATA.read_text());assert digest(PREDICTIONS)==data['predictions_sha256']
    checkpoint_sha=digest(checkpoint);same_checkpoint=checkpoint_sha==data['checkpoint_sha256']
    if Path(out).exists():raise FileExistsError(out)
    torch.set_num_threads(1);saved=torch.load(checkpoint,map_location='cpu',weights_only=True);saved.pop('optimizers',None)
    assert saved['step']==saved['args']['steps']
    for name,sha in saved['source_hashes'].items():assert digest(name)==sha
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'));del saved
    runner=PrefillInference(model,'sparse',prefill_chunk=128,grouped_gqa=True);seeds={};results=[]
    def generate(ids,prefill,decoder=None):
        decoder=decoder or runner
        cache={};start=time.perf_counter();last=prefill(ids,cache);first=last.clone();out=[[] for _ in ids]
        for step in range(64):
            tokens=last.argmax(-1)
            for values,token in zip(out,tokens.tolist()):values.append(token)
            if step!=63:last=decoder(tokens[:,None],cache)
        return out,first,time.perf_counter()-start
    with torch.inference_mode():
        for rows in data['cases']:
            ids=torch.tensor([r['input_ids'] for r in rows]);expected=[r['output_ids'] for r in rows]
            cold,baseline,cold_seconds=generate(ids,runner)
            if same_checkpoint:assert cold==expected
            else:expected=cold
            reference_error=None
            if reference:
                recurrent=Inference(model,'sparse',prefill_chunk=128)
                original,ref_logits,_=generate(ids,recurrent,recurrent);assert original==cold
                torch.testing.assert_close(ref_logits,baseline,atol=5e-4,rtol=1e-4)
                reference_error=float((ref_logits-baseline).abs().max())
            tags={r['subject'] for r in rows};prefill=runner;seed_seconds=0.;prefix_tokens=0
            if len(tags)==1:
                tag=next(iter(tags));prefix=data['prefixes'][tag];prefix_tokens=len(prefix)
                if prefix:
                    if tag not in seeds:
                        start=time.perf_counter();seeds[tag]=PrefixInference(runner,prefix);seed_seconds=time.perf_counter()-start
                    prefill=seeds[tag]
            cached,actual,cached_seconds=generate(ids,prefill);assert cached==expected
            torch.testing.assert_close(actual,baseline,atol=5e-4,rtol=1e-4)
            results.append(dict(batch_start=rows[0]['batch_start'],items=len(rows),prompt_tokens=ids.shape[1],prefix_tokens=prefix_tokens,
                seed_seconds=seed_seconds,cold_seconds=cold_seconds,cached_seconds=cached_seconds,
                prefill_logit_max_abs_error=float((actual-baseline).abs().max()),all_64_original_ids_exact=True if same_checkpoint else None,
                all_64_cached_ids_equal_cold=True,recurrent_logit_max_abs_error=reference_error))
    report=dict(predictions_sha256=data['predictions_sha256'],checkpoint_sha256=checkpoint_sha,results=results,
        all_original_ids_exact=True if same_checkpoint else None,all_cached_ids_equal_cold=True,
        all_recurrent_ids_equal_cold=True if reference else None,probe_sha256=digest(__file__),
        selection='Prior prepared fixture uses full 64-token HF records and batch shapes; equivalence only, not model accuracy',note='Warm in-process timings, limited cases. No full throughput or GPU kernel claim.')
    Path(out).write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--checkpoint',default='runs/qwen-sparse/last.pt');parser.add_argument('--out',default='runs/mmmlu-prefix-trained-proof.json');parser.add_argument('--reference',action='store_true');a=parser.parse_args()
    prepare() if a.prepare else check(a.checkpoint,a.out,a.reference)
