"""Replay a real eight-item batch, including a three-record partial restart."""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate_batched as engine
from eval_tasks import Task, correct
from prefix_inference import common_prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/eval-sparse-supergpqa-full.jsonl')
    parser.add_argument('--prefix',default='runs/retention-proof')
    parser.add_argument('--report',default='runs/retention-trained-proof.json')
    args=parser.parse_args();source = Path(args.source)
    if Path(args.report).exists():raise FileExistsError(args.report)
    group = []
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if group and row['batch_start'] != group[0]['batch_start']:group = []
            group.append(row)
            if len(group) == 8:break
    assert len(group) == 8 and all(r['batch_items']==8 and len(r['output_ids'])==64 for r in group)
    assert len(common_prefix([r['input_ids'] for r in group],128)) == 1280
    adapter = SimpleNamespace(__file__=__file__, correct=correct,
        load_tasks=lambda:[Task('supergpqa',5,True,'choice',[(r['prompt'],r['gold']) for r in group])])
    actual_import = engine.importlib.import_module
    def run(out, resume=False):
        argv = ['evaluate_batched.py','runs/qwen-sparse/last.pt','--out',str(out),'--tasks','supergpqa','--full',
                '--tokens','64','--context','8192','--batch-size','8','--prefill-chunk','128','--gdn-scan','chunk',
                '--prefix-cache','--grouped-gqa','--device','cpu','--seconds','60']
        if resume:argv.append('--resume')
        with patch.object(sys,'argv',argv),patch.object(engine.importlib,'import_module',
            side_effect=lambda n:adapter if n=='eval_tasks' else actual_import(n)),contextlib.redirect_stdout(io.StringIO()):
            engine.main()
    full=Path(args.prefix+'-full.jsonl');partial=Path(args.prefix+'-partial.jsonl')
    if full.exists() or partial.exists():raise FileExistsError('Keep existing proof outputs')
    run(full)
    lines=full.read_text().split('\n');records=[json.loads(s) for s in lines if s]
    assert [r['output_ids'] for r in records] == [r['output_ids'] for r in group]
    partial.write_text('\n'.join(lines[:3])+'\n')
    meta=json.loads(Path(str(full)+'.meta.json').read_text());meta.update(status='time_budget',completed_items=3,full_dataset_generated=False,full_dataset_attempted=False)
    count=meta['tasks']['supergpqa'];count.update(n=3,scored=3,correct=sum(r['correct'] for r in records[:3]))
    Path(str(partial)+'.meta.json').write_text(json.dumps(meta))
    run(partial,True)
    with partial.open() as stream:resumed=[json.loads(s) for s in stream]
    assert [(r['output_ids'],r['prediction']) for r in resumed] == [(r['output_ids'],r['prediction']) for r in records]
    result=dict(engine_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),items=8,new_tokens_per_item=64,
        all_original_ids_exact=True,partial_resume_exact=True,partial_prefix_records=3,
        original_batch_start=group[0]['batch_start'],original_indices=[r['index'] for r in group],
        note='Actual step8000 checkpoint; unchanged chunk/prefix/grouped-GQA CPU1 math. Eight cases, not universal numerical equivalence.')
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
