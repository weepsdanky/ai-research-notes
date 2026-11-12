"""Bind completed Mac control code predictions to the existing isolated judges."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate_ngram_placement import check_outputs, training_input
from qwen35_quality_data import adapter_for, digest, load, prepare_jobs, write_json

PROTOCOL = Path('profiles/qwen35_quality_protocol.json')


def attach_prompt(row, expected):
    task,index,prompt,gold,ids = expected
    assert (row['task'],row['index'],row['gold'],row['input_ids'])==(task.name,index,gold,ids)
    assert row['correct'] is None
    attached = dict(prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=len(ids))
    for key,value in attached.items():
        assert key not in row or row[key]==value
    return dict(row,**attached)


def export(args):
    from tokenizers import Tokenizer

    predictions = args.predictions
    original = load(str(predictions)+'.meta.json')
    receipt = load(args.generation_audit)
    identity = original['identity']
    assert original['status']=='complete' and original['identity']==receipt['identity']
    assert receipt['complete'] and receipt['output_sha256']==digest(predictions)
    assert identity.get('family')==args.experiment and identity['arm']==args.arm
    assert identity['job'] in ('code','multipl') and identity['decoding']=='greedy'
    assert identity['protocol_sha256']==digest(PROTOCOL)
    assert identity['training_audit_sha256']==digest(args.training_audit)
    assert all(digest(path)==sha for path,sha in identity['sources'].items())
    profile, arm, checkpoint, _ = training_input(args)
    assert arm['checkpoint_sha256']==identity['checkpoint_sha256']==digest(checkpoint)
    job, = [item for item in load(PROTOCOL)['jobs'] if item['name']==identity['job']]
    assert identity['job_spec']==job
    adapter,tasks = adapter_for(job)
    tokenizer_path = '.cache/references/tokenizer.json'
    assert digest(tokenizer_path)==identity['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(tokenizer_path)
    rows, content, _ = prepare_jobs(tasks,tokenizer,job)
    assert content==identity['content_sha256']
    stops = {task.name:adapter.stopping(task.name) for task in tasks}
    assert stops==identity['stop_strings']
    counts, _, _ = check_outputs(predictions,rows,adapter,tokenizer,stops,job,profile['config'])
    assert counts==dict(items=job['target_items'],scored=0,correct=0)
    assert receipt['items']==receipt['target']==len(rows) and receipt['scored']==0

    meta = Path(str(args.out)+'.meta.json')
    assert not meta.exists() and not args.out.exists()
    temporary = args.out.with_suffix('.tmp')
    assert not temporary.exists()
    with predictions.open() as source, temporary.open('x') as out:
        for line,expected in zip(source,rows,strict=True):
            out.write(json.dumps(attach_prompt(json.loads(line),expected),ensure_ascii=False)+'\n')
    temporary.replace(args.out)
    sources = dict(identity['sources'])
    sources['export_control_code.py']=digest(__file__)
    judge_identity = dict(suite=job['suite'],tokens=job['tokens'],context=job['context'],
                          checkpoint_sha256=identity['checkpoint_sha256'],tokenizer_sha256=identity['tokenizer_sha256'],
                          content_sha256=content,stop_strings=stops,source_hashes=sources,
                          original_control_identity=identity)
    write_json(meta,dict(status='complete',full_dataset_generated=True,completed_items=len(rows),
                        target_items=len(rows),scoring_status='pending official code harness',
                        identity=judge_identity,export_sha256=digest(args.out),
                        original_predictions=str(predictions),original_predictions_sha256=digest(predictions),
                        original_audit=str(args.generation_audit),original_audit_sha256=digest(args.generation_audit)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('table5',), help='Omit for Table 7 placement controls')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--training-audit', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--generation-audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.set_defaults(capacity_family=None)
    args=parser.parse_args()
    assert args.out.parent.resolve()==Path('runs').resolve()
    export(args)


if __name__=='__main__':
    main()
