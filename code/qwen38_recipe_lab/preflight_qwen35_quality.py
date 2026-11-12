"""Preflight all complete public quality jobs with Base IDs; no model import."""
import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
from tokenizers import Tokenizer
from qwen35_quality_data import PROTOCOL, load, digest, adapter_for, prepare_jobs, batches, write_json
from run_dense_text_guarded import pressure


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    assert not Path(args.out).exists()
    protocol=load(PROTOCOL);assert digest(protocol['tokenizer']['file'])==protocol['tokenizer']['sha256']
    tokenizer=Tokenizer.from_file(protocol['tokenizer']['file']);tick=time.monotonic()
    result=dict(status='running',protocol_sha256=digest(PROTOCOL),source_sha256=digest(__file__),jobs=[],scope='Complete source preparation only, no model generation/quality score; capacity failures preserve the protocol and all source rows')
    for job in protocol['jobs']:
        if pressure()!=1:result['status']='memory_pressure';break
        start=time.monotonic();adapter,tasks=adapter_for(job)
        try:rows,content,prefixes=prepare_jobs(tasks,tokenizer,job)
        except ValueError as error:
            result['jobs'].append(dict(name=job['name'],status='capacity_refused',error=str(error),target=job['target_items']))
        else:
            sizes={size:0 for size in range(1,9)}
            for batch in batches(rows,8):sizes[len(batch)]+=1
            result['jobs'].append(dict(name=job['name'],status='passed_complete_source_preflight',items=len(rows),content_sha256=content,max_prompt_tokens=max(len(row[4]) for row in rows),required_context=max(len(row[4]) for row in rows)+job['tokens'],declared_context=job['context'],prefix_lengths={key:len(value) for key,value in prefixes.items()},batch_sizes=sizes,seconds=time.monotonic()-start))
            del rows,prefixes
        del tasks,adapter;gc.collect()
        result.update(seconds=time.monotonic()-tick,torch_loaded='torch' in sys.modules,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(args.out,result)
        print(json.dumps({key:result['jobs'][-1][key] for key in ['name','status']}),flush=True)
    else:result['status']='complete' if all(row['status']=='passed_complete_source_preflight' for row in result['jobs']) else 'capacity_review_required'
    assert 'torch' not in sys.modules
    write_json(args.out,result)


if __name__=='__main__':main()
