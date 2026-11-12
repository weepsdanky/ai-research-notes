"""Eight paper languages, complete locked public MultiPL-E configurations.

Python is derived with the pinned publisher translator/cleaned originals;
it has 161 tasks, distinct from EvalPlus's original 164-task Python set.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import pyarrow.parquet as pq
from eval_tasks import Task


@lru_cache
def datasets():
    result = {}
    for spec in json.loads(Path('multipl_sources.lock.json').read_text()):
        path = Path('data/multipl')/spec['name']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError('MultiPL-E source drift')
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.suffix == '.jsonl' else pq.read_table(path).to_pylist()
        if len({r['name'] for r in rows}) != len(rows):raise ValueError('Duplicate MultiPL-E problem')
        result[spec['language']] = rows
    if set(result) != {'py', 'cpp', 'java', 'php', 'ts', 'cs', 'sh', 'js'}:
        raise ValueError('All eight paper languages required')
    return result


def load_tasks():
    for lang, rows in datasets().items():
        yield Task('multipl-'+lang, 0, False, 'code_pending', [(r['prompt'], r['name']) for r in rows])


def stopping(task):
    rows = datasets()[task.removeprefix('multipl-')]
    stops = rows[0]['stop_tokens']
    if any(r['stop_tokens'] != stops for r in rows):raise ValueError('Expected consistent publisher language stop tokens')
    return stops


def completion(raw, stops):
    end = min([raw.index(stop) for stop in stops if stop in raw] or [len(raw)])
    return raw[:end]


def correct(prediction, gold, kind):
    if kind != 'code_pending':raise ValueError('Code must be judged separately')
    return None
