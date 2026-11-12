"""Limited same-weight dense/QSA CPU prefill/decode timing; not a kernel claim."""
import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import time
import torch
from model import Model, Config
from prefill_inference import PrefillInference
from audit_stage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('--out',required=True)
    parser.add_argument('--seconds',type=int,default=1200)
    parser.add_argument('--grouped-gqa',action='store_true',help='Current optimized grouped path applies to sparse attention only')
    args=parser.parse_args()
    if Path(args.out).exists():raise FileExistsError(args.out)
    assert 1<=args.seconds<=1200
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert saved['step']==saved['args']['steps']==8000
    for file,sha in saved['source_hashes'].items():assert digest(file)==sha
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved['model']);del saved
    fixture=Path('runs/mmmlu-prefix-cases.json');data=json.loads(fixture.read_text())
    cases=[data['cases'][0],data['cases'][2],max(data['cases'],key=lambda rows:len(rows[0]['input_ids']))]
    results=[];began=time.monotonic();complete=True

    def run(ids,stage):
        runner=PrefillInference(model,stage,prefill_chunk=128,grouped_gqa=args.grouped_gqa);cache={}
        tick=time.perf_counter();logits=runner(ids,cache);prefill=time.perf_counter()-tick
        tick=time.perf_counter();outputs=[]
        for step in range(64):
            tokens=logits.argmax(-1);outputs.append(tokens.tolist())
            if step!=63:logits=runner(tokens[:,None],cache)
        return dict(prefill_seconds=prefill,decode_63_steps_seconds=time.perf_counter()-tick,output_ids=outputs)

    with torch.inference_mode():
        for index,rows in enumerate(cases):
            ids=torch.tensor([row['input_ids'] for row in rows]);expected={}
            for stage in ('pretrain','sparse'):run(ids,stage)  # One warm-up per stage/shape.
            for trial in range(3):
                for stage in (('pretrain','sparse') if trial%2==0 else ('sparse','pretrain')):
                    if time.monotonic()-began>=args.seconds:
                        complete=False;break
                    row=run(ids,stage)
                    if stage in expected:assert row['output_ids']==expected[stage]
                    else:expected[stage]=row['output_ids']
                    row.update(case=index,stage=stage,trial=trial,batch_size=len(rows),prompt_tokens=ids.shape[1])
                    results.append(row)
                if not complete:break
            if not complete:break
    summaries=[]
    for case in range(len(cases)):
        for stage in ('pretrain','sparse'):
            rows=[r for r in results if r['case']==case and r['stage']==stage]
            if rows:summaries.append(dict(case=case,stage=stage,batch_size=rows[0]['batch_size'],prompt_tokens=rows[0]['prompt_tokens'],trials=len(rows),prefill_median_seconds=statistics.median(r['prefill_seconds'] for r in rows),decode_median_seconds=statistics.median(r['decode_63_steps_seconds'] for r in rows)))
    report=dict(checkpoint=args.checkpoint,checkpoint_sha256=digest(args.checkpoint),fixture_sha256=digest(fixture),complete=complete and len(results)==18,
        expected_trials=18,completed_trials=len(results),grouped_gqa=args.grouped_gqa,torch=str(torch.__version__),platform=platform.platform(),threads=1,
        process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        source_sha256={p:digest(p) for p in ('model.py','inference.py','prefill_inference.py','reference.py',__file__)},summaries=summaries,results=results,
        scope='Three fixed real input batches; warm weights, fresh inference cache, three trials, alternating order. Dense uses the same weights with pretrain attention; QSA includes its indexer/gather. Always64 tokens including63 decode forwards, without EOS stopping: timing only, not quality scoring.',
        limitations='Includes GR/MoE/vocabulary head and Python eager overhead. Grouped GQA optimization is sparse-only when enabled; ungrouped sparse reference prefill also computes unused KL. These are implemented whole-path timings, not isolated algorithm costs; setup/tokenization excluded. Cumulative process RSS is not a per-stage memory comparison. Other host apps are uncontrolled. No GPU sparse-kernel or full-suite throughput claim.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
