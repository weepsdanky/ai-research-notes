"""Prove complete source/ID generation coverage without executing candidate code."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from judge_multipl import validated_jobs
from audit_generation_sources import verify_generation_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',default='runs/eval-sparse-multipl-full.jsonl')
    parser.add_argument('--out',default='runs/multipl-generation-audit.json')
    parser.add_argument('--checkpoint',default='runs/qwen-sparse/last.pt')
    parser.add_argument('--source-archive',default='9de3240')
    args=parser.parse_args()
    path=Path(args.predictions);meta,data,jobs=validated_jobs(str(path))
    with Path(args.checkpoint).open('rb') as stream:checkpoint_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert checkpoint_sha==meta['identity']['checkpoint_sha256'], 'Checkpoint identity mismatch'
    archive=subprocess.run(['git','rev-parse',args.source_archive],capture_output=True,text=True,check=True).stdout.strip()
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows)==len(jobs)==len(data)==1277
    sources=verify_generation_sources(meta['identity'],archive)
    counts=Counter(r['task'] for r in rows);stops=Counter(r['stop'] for r in rows)
    examples=[]
    for name in counts:
        row=next(r for r in rows if r['task']==name)
        examples.append(dict(task=name,index=row['index'],problem=row['gold'],generated_tokens=len(row['output_ids']),raw_excerpt=row['raw_prediction'][:160],correctness=None))
    report=dict(at=datetime.now().astimezone().isoformat(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),items=len(rows),
        tasks=dict(counts),stops=dict(stops),generated_tokens=sum(len(r['output_ids']) for r in rows),
        all_original_prompts_gold_names_input_ids_raw_decoding_and_stopped_completions_verified=True,
        generation_engine_source_archive=archive,generation_checkpoint_sha256=checkpoint_sha,verified_generation_sources=sources,predictions_path=str(path),all_correctness_pending=all(r['correct'] is None for r in rows),
        examples=examples,note='No candidate programs executed here. Full generation coverage does not establish any test pass rate.')
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))


if __name__=='__main__':main()
