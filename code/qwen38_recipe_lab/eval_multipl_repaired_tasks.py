"""Three isolated repaired fixtures, explicitly separate from the public protocol."""
import hashlib
import json
from pathlib import Path
from eval_tasks import Task
from eval_multipl_tasks import completion


def datasets():
    data=Path('data/multipl-repaired/repairs.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest()!=json.loads(Path('multipl_repairs.lock.json').read_text())['data_sha256']:raise ValueError('Repair data drift')
    return [json.loads(s) for s in data.decode().splitlines()]


def load_tasks():
    for row in datasets():yield Task('multipl-repaired-'+row['language'],0,False,'code_pending',[(row['prompt'],row['name'])])


def stopping(task):return next(r['stop_tokens'] for r in datasets() if task=='multipl-repaired-'+r['language'])


def correct(prediction,gold,kind):return None
