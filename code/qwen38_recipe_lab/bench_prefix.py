"""Real-checkpoint prefix reuse proof; timings may include concurrent CPU load."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Model,Config
from prefill_inference import PrefillInference
from prefix_inference import PrefixInference,common_prefix
from eval_extra_tasks import load_tasks


def main():
    out=Path('runs/prefix-trained-proof.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);groups=defaultdict(list)
    for line in Path('runs/eval-sparse-gsm8k-full.jsonl').open():
        row=json.loads(line);groups[row['batch_start']].append(row)
    rows=next(rows for rows in groups.values() if len(rows)==8)
    cases=[('gsm8k',rows)]
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json')
    for task in load_tasks():
        if task.name not in ('math-geometry','math-precalculus'):continue
        bins=defaultdict(list)
        for i,(prompt,gold) in enumerate(task.rows):
            ids=tokenizer.encode(prompt,add_special_tokens=False).ids
            bins[len(ids)].append(dict(task=task.name,index=i,input_ids=ids))
        cases.append((task.name,max(bins.values(),key=len)[:8]))
    saved=torch.load('runs/qwen-sparse/last.pt',map_location='cpu',weights_only=True);saved.pop('optimizers',None)
    for name,sha in saved['source_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'));del saved
    results=[]
    with torch.inference_mode():
        for name,rows in cases:
            ids=torch.tensor([r['input_ids'] for r in rows]);prefix=common_prefix(ids.tolist(),128)
            runner=PrefillInference(model,'sparse',prefill_chunk=128);cached=PrefixInference(runner,prefix)
            expected=[];outputs=[];timings=[]
            for method in (runner,cached):
                cache={};tick=time.perf_counter();logits=method(ids,cache);prefill=time.perf_counter()-tick;generated=[]
                for step in range(64):
                    token=logits.argmax(-1);generated.append(token.tolist())
                    if step!=63:logits=method(token[:,None],cache)
                value=list(map(list,zip(*generated)))
                if method is runner:expected=value
                else:outputs=value
                timings.append(dict(prefill_seconds=prefill,total_seconds=time.perf_counter()-tick))
            original=all(row['output_ids'][:64]==value for row,value in zip(rows,outputs)) if name=='gsm8k' else None
            results.append(dict(task=name,items=len(rows),prefix_tokens=len(prefix),prompt_tokens=ids.shape[1],exact_baseline_ids=outputs==expected,
                exact_previous_gsm_ids=original,source_items=[dict(task=r['task'],index=r['index']) for r in rows],output_ids=outputs,diagnostic_times=timings))
    report=dict(results=results,all_exact=all(r['exact_baseline_ids'] and r['exact_previous_gsm_ids'] is not False for r in results),
        source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['prefix_inference.py','prefill_inference.py','inference.py','model.py']},
        note='CPU proof during another native model generation; diagnostic times are not controlled benchmark speedups. Prefix construction cost is separate and amortized only when reused.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'}));assert report['all_exact']


if __name__=='__main__':main()
