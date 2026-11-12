"""Combine an immutable complete evaluation with exact capacity-only retries.

Keeps both source files, hashes and per-record origin. Never replaces a generated
answer with a more favorable rerun or hides a changed prompt/model/token budget.
"""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original');p.add_argument('retry');p.add_argument('--out',required=True);a=p.parse_args()
    metas=[json.loads(Path(f+'.meta.json').read_text()) for f in (a.original,a.retry)]
    if any(m['status']!='complete' for m in metas):raise ValueError('Both source runs must be complete')
    for key in ('checkpoint_sha256','runtime_model_sources','tokens'):
        if metas[0]['identity'][key]!=metas[1]['identity'][key]:raise ValueError('Different model or decode budget')
    if metas[1]['identity']['retry_source_sha256']!=sha(a.original):raise ValueError('Retry provenance mismatch')
    retries={}
    for line in Path(a.retry).open():
        row=json.loads(line);key=(row['task'],row['index'])
        if key in retries or row['stop']=='context_capacity' or not row['output_ids']:raise ValueError('Invalid retry result')
        retries[key]=row
    counts=defaultdict(lambda:dict(n=0,correct=0,capacity_failures=0));used=set();total=0
    with Path(a.out).open('x') as stream:
        for line in Path(a.original).open():
            row=json.loads(line);origin=a.original;key=(row['task'],row['index'])
            if row['stop']=='context_capacity':
                candidate=retries[key]
                for field in ('task','index','prompt','prompt_sha256','input_ids','gold','shots','cot','answer_kind'):
                    if candidate[field]!=row[field]:raise ValueError(f'Changed retry input: {field}')
                row=candidate;origin=a.retry;used.add(key)
            elif key in retries:raise ValueError('Refuse replacing an already-generated answer')
            row=dict(row,evidence_origin=origin)
            count=counts[row['task']];count['n']+=1;count['correct']+=int(row['correct']);count['capacity_failures']+=int(row['stop']=='context_capacity')
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');total+=1
    if used!=set(retries):raise ValueError('Unused retry entries')
    for name,count in counts.items():
        count['total_dataset']=metas[0]['tasks'][name]['total_dataset'];count['accuracy']=count['correct']/count['n']
        if count['n']!=count['total_dataset']:raise ValueError('Original source did not cover full selected datasets')
    report=dict(status='complete',completed_items=total,full_dataset_attempted=True,full_dataset_generated=not any(v['capacity_failures'] for v in counts.values()),
        sources=[dict(path=f,sha256=sha(f),identity=m['identity']) for f,m in zip((a.original,a.retry),metas,strict=True)],
        tasks=dict(counts),macro_accuracy=sum(v['accuracy'] for v in counts.values())/len(counts),
        micro_accuracy=sum(v['correct'] for v in counts.values())/total,
        note='Composite actual generations; same untruncated inputs/model/token budget, different recorded cache and context-cap implementations; not official Qwen prompts or scores')
    other=[v for k,v in counts.items() if k!='mgsm-en']
    if all(k.startswith('mgsm-') for k in counts):report['macro_non_english']=sum(v['accuracy'] for v in other)/len(other)
    Path(a.out+'.meta.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report['tasks'],indent=2))


if __name__=='__main__':main()
