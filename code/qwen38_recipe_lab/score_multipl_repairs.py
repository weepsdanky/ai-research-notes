"""Score an explicitly non-official repaired variant, preserving all raw results."""
import argparse
from datetime import datetime
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from tokenizers import Tokenizer
from eval_multipl_repaired_tasks import datasets,completion,stopping
from multipl_judge import execute
from sandbox import Sandbox


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',default='runs/eval-sparse-multipl-repaired.jsonl')
    parser.add_argument('--checkpoint',default='runs/qwen-sparse/last.pt')
    parser.add_argument('--raw',default='runs/multipl-raw-full.jsonl')
    parser.add_argument('--scaffolds',default='runs/multipl-scaffolds-full-v2.jsonl')
    parser.add_argument('--controls',default='runs/multipl-repaired-controls.json')
    parser.add_argument('--scores-out',default='runs/multipl-repaired-score.json')
    parser.add_argument('--out',default='runs/multipl-final-audit.json')
    parser.add_argument('--deadline',required=True,help='Aware round deadline; reserve enough for three 90-second tasks and cleanup')
    args=parser.parse_args()
    if Path(args.out).exists() or Path(args.scores_out).exists():raise FileExistsError('Preserve existing outputs; choose fresh explicit result paths')
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or (deadline-datetime.now().astimezone()).total_seconds()<500:parser.error('At least500 seconds required before deadline; no sandbox started')
    path=Path(args.predictions);meta=json.loads(Path(str(path)+'.meta.json').read_text())
    assert meta['status']=='complete' and meta['full_dataset_generated']
    checkpoint_sha=digest(args.checkpoint)
    assert meta['identity']['checkpoint_sha256']==checkpoint_sha
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json');data={r['language']:r for r in datasets()}
    jobs={}
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);lang=row['task'].removeprefix('multipl-repaired-');problem=data[lang];key=lang,problem['name']
            assert key not in jobs and row['gold']==problem['name'] and row['prompt']==problem['prompt'] and row['correct'] is None
            assert row['input_ids']==tokenizer.encode(problem['prompt'],add_special_tokens=False).ids
            assert 0<len(row['output_ids'])<=1024 and row['raw_prediction']==tokenizer.decode(row['output_ids'],skip_special_tokens=False)
            assert row['prediction']==completion(tokenizer.decode(row['output_ids'],skip_special_tokens=True),stopping(row['task']))
            jobs[key]=(problem,row['prediction'])
    with Path(args.raw).open() as stream:original=[json.loads(line) for line in stream]
    with Path(args.scaffolds).open() as stream:scaffold_rows=[json.loads(line) for line in stream]
    scaffold={(r['language'],r['name']):r for r in scaffold_rows};raw={(r['language'],r['name']):r for r in original}
    assert len(original)==len(scaffold_rows)==len(raw)==len(scaffold)==1277 and raw.keys()==scaffold.keys()
    invalid={key for key,row in scaffold.items() if not row['scaffold_valid']};assert invalid==jobs.keys() and len(jobs)==3
    raw_meta=json.loads(Path(args.raw+'.meta.json').read_text())
    assert raw_meta['status']=='complete' and raw_meta['not_model_accuracy']
    assert checkpoint_sha==raw_meta['identity']['generation_identity']['checkpoint_sha256']
    assert meta['identity']['tokens']==raw_meta['identity']['generation_identity']['tokens']==1024
    controls=json.loads(Path(args.controls).read_text());assert len(controls)==6
    assert all(r['passed']==(r['control']=='canonical') for r in controls)
    if (deadline-datetime.now().astimezone()).total_seconds()<500:parser.error('Insufficient round budget after preflight; no sandbox started')
    sandbox=Sandbox('multipl');assert all(r['image_id']==sandbox.image for r in original)
    repaired={};results=[]
    for (lang,name),(problem,body) in jobs.items():
        result=execute(sandbox,lang,problem,body);result.update(language=lang,name=name)
        repaired[lang,name]=result;results.append(result)
        Path(args.scores_out).write_text(json.dumps(results,indent=2)+'\n')
        print(lang,result['status'],result['passed'],flush=True)
    assert all(r['image_id']==sandbox.image for r in results)
    counts=defaultdict(lambda:dict(n=0,correct=0,repaired_fixtures=0))
    for key,row in raw.items():
        selected=repaired[key] if key in invalid else row;count=counts[key[0]]
        count['n']+=1;count['correct']+=int(selected['passed']);count['repaired_fixtures']+=int(key in invalid)
    assert len(counts)==8
    report=dict(not_official_protocol=True,scope='Mac repaired composite: 1274 original valid scaffolds plus exactly 3 independently identified/re-generated repaired fixtures; no answer selection',
        original_raw_tested=1277,original_raw_passes=sum(r['passed'] for r in original),original_ungradable_scaffolds=[list(key) for key in sorted(invalid)],
        original_valid_scaffolds=1277-len(invalid),original_valid_correct=sum(row['passed'] for key,row in raw.items() if key not in invalid),
        repaired_composite_items=1277,tasks=dict(counts),mac_repaired_eight_language_macro=sum(c['correct']/c['n'] for c in counts.values())/8,
        original_full_model_accuracy=None,note='Original public protocol cannot establish a full model accuracy because three fixtures fail before candidate invocation. Repairs change C++/PHP prompts; engines are versioned. This is a separate Mac variant, not an official Qwen score.',
        source_sha256={f:digest(f) for f in ['score_multipl_repairs.py','multipl_repairs.lock.json','multipl_judge.py','multipl_driver.py','sandbox.py','vm.sh',args.raw,args.scaffolds,str(path),args.controls]})
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
