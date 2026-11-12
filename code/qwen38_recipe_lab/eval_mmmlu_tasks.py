"""Complete cached MMMLU language files; explicit English MMLU dev 5-shot.

The publisher supplies translated test CSVs, not translated dev exemplars.
Qwen's private translated prompts/examples are unknown. Never synthesize them.
"""
import csv
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from eval_tasks import Task, correct

SUBJECT_ALIASES = {
    'college_mathematics_test.csv_sw-KE.csv': 'college_mathematics',
    'security_studies_test-sw-KE.csv': 'security_studies',
    'college_mathematics_test.csv_YO-NG.csv': 'college_mathematics',
}


def validate_selection(tasks):
    if len(tasks) != 1:
        raise ValueError('Run one complete MMMLU language at a time to bound tokenized prompt memory')


def load_tasks():
    root = Path('data/mmmlu')
    sources = {s['name']: s for s in json.loads(Path('eval_sources.lock.json').read_text())}
    exemplars = {}
    for spec in json.loads(Path('mmmlu_sources.lock.json').read_text()):
        file = root/spec['name']
        if not file.exists():continue
        with file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError('MMMLU cache drift')
        with file.open(encoding='utf-8-sig', newline='') as stream:rows = list(csv.DictReader(stream))
        if len(rows) != 14042:raise ValueError('Expected the complete 14042-item language test')
        prompts = [];prefix_groups=[]
        for row in rows:
            subject = SUBJECT_ALIASES.get(row['Subject'], row['Subject'])
            if subject not in exemplars:
                name = f'mmlu-{subject}-dev.parquet';path = Path('data/eval-source')/name
                if name not in sources:raise ValueError(f'Unknown MMMLU source subject: {row["Subject"]}')
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != sources[name]['sha256']:
                        raise ValueError('English exemplar drift')
                dev = pq.read_table(path).to_pylist()
                if len(dev) != 5:raise ValueError('Expected five published English dev examples')
                exemplars[subject] = '\n\n'.join(r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(r['choices']))+'\nAnswer: '+chr(65+r['answer']) for r in dev)
            if row['Answer'] not in 'ABCD' or len(row['Answer']) != 1:raise ValueError('Invalid translated choice label')
            question = row['Question']+'\n'+'\n'.join(f'{letter}. {row[letter]}' for letter in 'ABCD')+'\nAnswer:'
            prompts.append((exemplars[subject]+'\n\n'+question, row['Answer']));prefix_groups.append(subject)
        task=Task('mmmlu-'+spec['language'].lower().replace('-', '_'), 5, False, 'choice', prompts)
        task.prefix_groups=prefix_groups
        yield task
