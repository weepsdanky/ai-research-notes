"""Recheck complete core generations against source prompts, IDs and scoring."""
import argparse
import hashlib
import importlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tokenizers import Tokenizer


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',nargs='+');parser.add_argument('--out',required=True)
    parser.add_argument('--suite',choices=['core','extra','mmmlu'],default='core')
    parser.add_argument('--allow-partial',action='store_true',help='Audit a terminal saved prefix and label incomplete coverage explicitly')
    args = parser.parse_args();adapter=importlib.import_module({'core':'eval_tasks','extra':'eval_extra_tasks','mmmlu':'eval_mmmlu_tasks'}[args.suite])
    tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
    result = dict(at=now(),scorer_source_sha256=digest(adapter.__file__),files={})
    for name in args.predictions:
        path = Path(name);meta = json.loads(Path(name+'.meta.json').read_text())
        tasks = {t.name:t for t in adapter.load_tasks() if t.name in meta['tasks']}
        if meta['status'] not in ('complete','time_budget','interrupted_memory_pressure','memory_pressure','interrupted_signal','resource_monitor_failed'):raise ValueError('Terminal saved evaluation required')
        if not args.allow_partial and not meta['full_dataset_generated']:raise ValueError('Complete genuine generation required')
        seen = set();scores = defaultdict(lambda:dict(n=0,correct=0));examples = []
        # JSONL uses LF; stream without splitting Unicode separators in strings.
        for row in records(path):
            task = tasks[row['task']];key = row['task'],row['index']
            if key in seen:raise ValueError('Duplicate result')
            seen.add(key);prompt,gold = task.rows[row['index']]
            if (prompt,gold,task.kind,task.shots,task.cot)!=(row['prompt'],row['gold'],row['answer_kind'],row['shots'],row['cot']):raise ValueError('Source prompt/gold drift')
            if row['input_ids']!=tokenizer.encode(prompt,add_special_tokens=False).ids:raise ValueError('Input IDs drift')
            if not row['output_ids'] or any(tokenizer.id_to_token(i) is None for i in row['output_ids']):raise ValueError('Invalid generated IDs')
            decoded = tokenizer.decode(row['output_ids'],skip_special_tokens=False)
            if row['raw_prediction']!=decoded or row['prediction']!=decoded:raise ValueError('Raw output decoding drift')
            score = adapter.correct(decoded,gold,task.kind)
            if row['correct']!=score:raise ValueError('Score drift')
            count = scores[task.name];count['n'] += 1;count['correct'] += int(score)
            if len(examples)<3:examples.append({k:row[k] for k in ['task','index','gold','raw_prediction','correct']})
        expected = {(n,i) for n in meta['tasks'] for i in range(len(tasks[n].rows))}
        if not seen<=expected or not args.allow_partial and seen!=expected:raise ValueError('Unexpected/incomplete public task coverage')
        for n,c in scores.items():
            if (c['n'],c['correct'])!=(meta['tasks'][n]['n'],meta['tasks'][n]['correct']):raise ValueError('Meta count drift')
        total = sum(c['n'] for c in scores.values());right = sum(c['correct'] for c in scores.values())
        audit = dict(file_sha256=digest(path),terminal_status=meta['status'],items=total,target_items=len(expected),full_dataset_generated=seen==expected,
                     coverage='full' if seen==expected else 'partial generated prefix',correct=right,micro_accuracy=right/total if total else None,tasks=dict(scores),
                     all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified=True,examples=examples)
        if scores and seen==expected and all(n.startswith('bbh-') for n in scores):
            families = defaultdict(list)
            for n,c in scores.items():
                family = 'logical_deduction' if n.startswith('bbh-logical_deduction_') else 'tracking_shuffled_objects' if n.startswith('bbh-tracking_shuffled_objects_') else n.removeprefix('bbh-')
                families[family].append(c['correct']/c['n'])
            audit['bbh_27_configuration_macro_accuracy'] = sum(c['correct']/c['n'] for c in scores.values())/len(scores)
            audit['bbh_23_family_macro_accuracy'] = sum(sum(v)/len(v) for v in families.values())/len(families)
            if (len(scores),len(families))!=(27,23):raise ValueError('BBH grouping drift')
        result['files'][name] = audit
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({p:{k:v for k,v in a.items() if k not in ('tasks','examples')} for p,a in result['files'].items()},indent=2))


if __name__=='__main__':main()
