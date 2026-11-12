"""EvalPlus public base completion prompts; correctness requires its harness."""
import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from eval_tasks import Task

WHEEL_SHA256 = 'cd601debb67419113d10ac5c3317689d847f27de5d8cf3837975f3cab571b75d'


def datasets():
    lock = {s['name']: s for s in json.loads(Path('eval_sources.lock.json').read_text())}
    for name, file in [('humaneval', 'HumanEvalPlus-v0.1.10.jsonl.gz'), ('mbpp', 'MbppPlus-v0.2.0.jsonl.gz')]:
        raw = (Path('data/eval-source')/file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != lock[file]['sha256']:
            raise ValueError('EvalPlus data drift')
        yield name, [json.loads(s) for s in gzip.decompress(raw).decode().splitlines()]


def load_tasks():
    for name, rows in datasets():
        yield Task('evalplus-'+name, 0, False, 'code_pending', [(r['prompt'], r['task_id']) for r in rows])


def correct(prediction, gold, kind):
    if kind != 'code_pending':
        raise ValueError('Expected pending code correctness')
    return None


def stopping(task):
    # Exact constants from the locked official 0.3.1 wheel, not an invented
    # completion heuristic. No model weights/providers are loaded here.
    path = Path('.cache/references/evalplus-0.3.1-py3-none-any.whl')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WHEEL_SHA256:
        raise ValueError('EvalPlus wheel drift')
    namespace = {}
    with zipfile.ZipFile(path) as archive:
        exec(compile(archive.read('evalplus/provider/utility.py'), 'evalplus/provider/utility.py', 'exec'), namespace)
    dataset = task.removeprefix('evalplus-')
    return namespace['EOS'] + namespace['extra_eos_for_direct_completion'](dataset)


def completion(raw, stops):
    end = min([raw.index(stop) for stop in stops if stop in raw] or [len(raw)])
    return raw[:end].replace('\t', '    ')
