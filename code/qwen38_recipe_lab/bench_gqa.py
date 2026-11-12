"""Isolated CPU comparison on fixed completed MATH generations, not GPU kernels."""
import argparse
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
import torch
from model import Model,Config
from prefill_inference import PrefillInference
from prefix_inference import PrefixInference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grouped',action='store_true');parser.add_argument('--out',required=True);args=parser.parse_args()
    if Path(args.out).exists():raise FileExistsError(args.out)
    torch.set_num_threads(1)
    meta=json.loads(Path('runs/eval-sparse-math-full.jsonl.meta.json').read_text());assert meta['full_dataset_generated']
    bins=defaultdict(list)
    for line in Path('runs/eval-sparse-math-full.jsonl').open():
        row=json.loads(line);bins[row['batch_start']].append(row)
    candidates=[rows for rows in bins.values() if len(rows)==8 and len({r['task'] for r in rows})==1 and all(len(r['output_ids'])==64 for r in rows)]
    cases=[min(candidates,key=lambda rows:abs(rows[0]['prompt_tokens']-length)) for length in (1000,1800,2600)]
    assert len({rows[0]['batch_start'] for rows in cases})==3
    saved=torch.load('runs/qwen-sparse/last.pt',map_location='cpu',weights_only=True);saved.pop('optimizers',None)
    for file,sha in saved['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'));del saved
    results=[];tick=time.monotonic()
    with torch.inference_mode():
        for rows in cases:
            prefix_length=meta['identity']['prefix_cache']['plans'][rows[0]['task']]['tokens']
            runner=PrefillInference(model,'sparse',prefill_chunk=128,grouped_gqa=args.grouped)
            start=time.perf_counter();cached=PrefixInference(runner,rows[0]['input_ids'][:prefix_length]);seed_seconds=time.perf_counter()-start
            cache={};ids=torch.tensor([r['input_ids'] for r in rows]);start=time.perf_counter();logits=cached(ids,cache);prefill=time.perf_counter()-start
            outputs=[[] for _ in rows]
            for step in range(64):
                tokens=logits.argmax(-1)
                for out,token in zip(outputs,tokens.tolist()):out.append(token)
                if step!=63:logits=runner(tokens[:,None],cache)
            exact=[out==r['output_ids'] for out,r in zip(outputs,rows)]
            results.append(dict(task=rows[0]['task'],batch_start=rows[0]['batch_start'],items=8,prompt_tokens=ids.shape[1],prefix_tokens=prefix_length,
                seed_seconds=seed_seconds,prefill_seconds=prefill,total_seconds=time.perf_counter()-start,exact_original_ids=exact,
                source_items=[dict(task=r['task'],index=r['index']) for r in rows],output_ids=outputs))
    report=dict(grouped_gqa=args.grouped,results=results,all_exact=all(all(r['exact_original_ids']) for r in results),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        generation_seconds=time.monotonic()-tick,source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['inference.py','prefill_inference.py','prefix_inference.py']},
        note='Fresh CPU process per mode, one thread; same 24 completed MATH inputs and 64 generated tokens. Single measurements, not production kernel speedups.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'}));assert report['all_exact']


if __name__=='__main__':main()
