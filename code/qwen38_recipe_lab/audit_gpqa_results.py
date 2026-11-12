"""Audit GPQA source identity, full inputs, raw outputs and author scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tokenizers import Tokenizer
import eval_gpqa_tasks as adapter
from audit_core_results import digest, now, records
from audit_generation_sources import verify_generation_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    prefix = subprocess.run(['git', 'rev-parse', '--show-prefix'], capture_output=True, text=True, check=True).stdout.strip()
    locked = subprocess.run(['git', 'show', args.archive+':'+prefix+'gpqa_sources.lock.json'], capture_output=True, check=True).stdout
    if locked != Path('gpqa_sources.lock.json').read_bytes():
        raise ValueError('GPQA lock differs from source archive')
    adapter.sources()  # verifies every pinned data/reference byte
    meta = json.loads(Path(args.predictions+'.meta.json').read_text())
    identity = meta['identity']
    sources = verify_generation_sources(identity, args.archive)
    if digest(adapter.__file__) not in identity['source_hashes'].values():
        raise ValueError('Loaded GPQA adapter differs from recorded source')
    if identity['tokens'] != 600 or identity['context'] != 8192 or identity['suite'] != 'gpqa':
        raise ValueError('GPQA declared protocol drift')
    if identity['checkpoint_sha256'] != '1b0127c8d93ae879f041f202e789623de837a8fe7e56a557d06da98aa3afcb5e' or digest('runs/qwen-sparse-paper/last.pt') != identity['checkpoint_sha256']:
        raise ValueError('Checkpoint drift')
    if digest('.cache/references/tokenizer.json') != identity['tokenizer_sha256']:
        raise ValueError('Tokenizer drift')
    if meta['status'] not in ('complete', 'time_budget', 'interrupted_memory_pressure', 'memory_pressure', 'interrupted_signal', 'resource_monitor_failed'):
        raise ValueError('Saved terminal run required')
    tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
    tasks = {task.name: task for task in adapter.load_tasks() if task.name in meta['tasks']}
    declared = json.loads(Path('profiles/gpqa_evaluation.json').read_text())['variants']
    if set(tasks) != set(declared) or any(len(tasks[name].rows) != size for name, size in declared.items()):
        raise ValueError('Declared four-configuration coverage drift')
    if set(identity['tasks']) != set(tasks) or identity['scope'] != 'full selected datasets':
        raise ValueError('Full selection identity drift')
    ordered = []
    for task in tasks.values():
        for i, (prompt, gold) in enumerate(task.rows):
            ordered.append((len(tokenizer.encode(prompt, add_special_tokens=False).ids), task.name, i, task, prompt, gold))
    content = hashlib.sha256()
    for _, name, i, task, prompt, gold in sorted(ordered, key=lambda row: row[:3]):
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        content.update(json.dumps((name, i, task.shots, task.cot, task.kind, prompt, gold, ids), ensure_ascii=False).encode())
    if content.hexdigest() != identity['content_sha256']:
        raise ValueError('Full dataset content identity drift')
    expected = {(name, i) for name, task in tasks.items() for i in range(len(task.rows))}
    seen = set()
    scores = {name: dict(n=0, correct=0) for name in tasks}
    for row in records(args.predictions):
        key = row['task'], row['index']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate item')
        seen.add(key)
        task = tasks[row['task']]
        prompt, gold = task.rows[row['index']]
        if (prompt, gold, task.kind, task.shots, task.cot) != (row['prompt'], row['gold'], row['answer_kind'], row['shots'], row['cot']):
            raise ValueError('Source prompt/gold drift')
        if row['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest() or row['input_ids'] != tokenizer.encode(prompt, add_special_tokens=False).ids:
            raise ValueError('Input hash/IDs drift')
        if not row['output_ids'] or len(row['output_ids']) > 600 or len(row['input_ids'])+600 > 8192 or any(tokenizer.id_to_token(i) is None for i in row['output_ids']):
            raise ValueError('Invalid generated IDs/context')
        decoded = tokenizer.decode(row['output_ids'], skip_special_tokens=False)
        if row['raw_prediction'] != decoded or row['prediction'] != decoded:
            raise ValueError('Raw decoding drift')
        correct = adapter.correct(decoded, gold, task.kind)
        if row['correct'] != correct:
            raise ValueError('Author scoring drift')
        scores[task.name]['n'] += 1
        scores[task.name]['correct'] += int(correct)
    if meta['completed_items'] != len(seen) or meta['full_dataset_generated'] != (seen == expected):
        raise ValueError('Coverage metadata drift')
    if not args.allow_partial and seen != expected:
        raise ValueError('Full declared task coverage required')
    for name, score in scores.items():
        if any(score[k] != meta['tasks'][name][k] for k in ('n', 'correct')):
            raise ValueError('Metadata counts drift')
        score['target_items'] = len(tasks[name].rows)
        score['accuracy'] = score['correct']/score['n'] if score['n'] else None
    result = dict(at=now(), source_archive=args.archive, auditor_sha256=digest(__file__), sources=sources,
                  gpqa_lock_sha256=hashlib.sha256(locked).hexdigest(), checkpoint_sha256=identity['checkpoint_sha256'],
                  file_sha256=digest(args.predictions), terminal_status=meta['status'],
                  items=len(seen), target_items=len(expected), full_dataset_generated=seen==expected,
                  all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified=True, tasks=scores,
                  reporting='Separate template/configuration scores; no combined author/clean accuracy. No benchmark text exported.')
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
