"""Actual checkpoint/token comparison of public chunk vs recurrent prefill."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from model import Model, Config
from inference import Inference
from prefill_inference import PrefillInference


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', required=True);a = p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    torch.set_num_threads(1)
    keys={(r['task'],r['index']) for r in json.loads(Path('runs/exact-batch-benchmark.json').read_text())['source_items']}
    short=[];longest=None
    for line in Path('runs/eval-sparse-mmlu-full.jsonl').open():
        row=json.loads(line)
        if (row['task'],row['index']) in keys:short.append(row)
        if longest is None or row['prompt_tokens']>longest['prompt_tokens']:longest=row
    groups=defaultdict(list)
    for line in Path('runs/eval-sparse-gsm8k-full.jsonl').open():
        row=json.loads(line);groups[row['batch_start']].append(row)
    candidates=[v for v in groups.values() if len(v)==8]
    medium=min(candidates,key=lambda g:abs(g[0]['prompt_tokens']-1200))
    saved=torch.load('runs/qwen-sparse/last.pt',map_location='cpu',weights_only=True);saved.pop('optimizers',None)
    assert saved['step']==saved['args']['steps']
    for file,expected in saved['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'));del saved
    results=[]
    with torch.inference_mode():
        for rows in [short,medium,[longest]]:
            assert len({len(r['input_ids']) for r in rows})==1
            for name,klass in [('recurrent',Inference),('chunk',PrefillInference)]:
                tick=time.perf_counter();cache={};runner=klass(model,'sparse',prefill_chunk=128)
                logits=runner(torch.tensor([r['input_ids'] for r in rows]),cache);prefill=time.perf_counter()-tick
                outputs=[[] for _ in rows]
                for step in range(64):
                    tokens=logits.argmax(-1)
                    for out,tok in zip(outputs,tokens.tolist()):out.append(tok)
                    if step!=63:logits=runner(tokens[:,None],cache)
                matches=[out==r['output_ids'][:64] for out,r in zip(outputs,rows)]
                result=dict(scan=name,items=len(rows),prompt_tokens=rows[0]['prompt_tokens'],prefill_seconds=prefill,
                            total_seconds=time.perf_counter()-tick,exact_original_ids=matches,
                            source_items=[dict(task=r['task'],index=r['index']) for r in rows])
                results.append(result);print(result,flush=True)
    report=dict(results=results,all_exact=all(all(r['exact_original_ids']) for r in results),source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['inference.py','prefill_inference.py','reference.py']},note='CPU public chunk reference/cache optimization, not fused QSA kernel acceleration')
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
