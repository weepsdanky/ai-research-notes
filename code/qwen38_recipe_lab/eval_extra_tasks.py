"""Additional public datasets; shot counts match paper, Mac choices explicit."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import yaml
from eval_tasks import Task, correct as choice_correct

ROOT = Path('data/eval-source')
MATH_REFERENCE_SHA256 = 'c4101b1f51a2bb65665194aecd9f761d668c0c248a0350f196c2950e87488527'


def boxed(text):
    """Last balanced boxed/fbox expression; malformed output is not an answer."""
    markers = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not markers:
        return None
    start = markers[-1].end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == '{') - (text[index] == '}')
        if depth == 0:
            return text[start:index]
    return None


@lru_cache
def math_equivalent():
    path = Path('.cache/references/math_equivalence.py')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MATH_REFERENCE_SHA256:
        raise ValueError('MATH reference drift')
    namespace = {}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace['is_equiv']


def correct(prediction, gold, kind):
    if kind != 'boxed_math':
        return choice_correct(prediction, gold, kind)
    actual = boxed(prediction)
    if actual is None:
        matches = re.findall(r'(?:final\s+)?answer\s*(?:is\s*:?|:)\s*(.+)', prediction, re.I)
        actual = matches[-1].strip().rstrip('.') if matches else None
    return actual is not None and bool(actual.strip()) and math_equivalent()(actual, gold)


def verified(name):
    spec = next(s for s in json.loads(Path('eval_sources.lock.json').read_text()) if s['name'] == name)
    path = ROOT/name
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
            raise ValueError(f'Evaluation source drift: {name}')
    return path


def include_examples(path):
    """Five examples disjoint from every test question in this language.

    Published validation overlaps test. When fewer than five remain, fill from
    fixed MMLU anatomy dev (English); record this Mac protocol difference.
    """
    test = pq.read_table(verified(path.name)).to_pylist()
    validation_name = path.name.replace('-test-', '-validation-')
    validation = pq.read_table(verified(validation_name)).to_pylist()
    norm = lambda text: ' '.join(text.split()).casefold()
    questions = {norm(r['question']) for r in test}
    examples, provenance, seen = [], [], set(questions)
    for index, row in enumerate(validation):
        question = norm(row['question'])
        if question not in seen and len(examples) < 5:
            examples.append(row);provenance.append(dict(source=validation_name, index=index, fallback=False));seen.add(question)
    if len(examples) < 5:
        name = 'mmlu-anatomy-dev.parquet'
        for index, row in enumerate(pq.read_table(verified(name)).to_pylist()):
            question = norm(row['question'])
            if question not in seen and len(examples) < 5:
                examples.append(dict(question=row['question'], answer=row['answer'],
                                     **{'option_'+chr(97+i): value for i, value in enumerate(row['choices'])}))
                provenance.append(dict(source=name, index=index, fallback=True));seen.add(question)
    if len(examples) != 5:
        raise ValueError('Cannot construct five disjoint public examples')
    return test, examples, provenance


def load_tasks():
    root = Path('data/eval-source')
    names = ('mmlu-pro-validation.parquet', 'mmlu-pro-test.parquet')
    locked = {s['name']: s for s in json.loads(Path('eval_sources.lock.json').read_text())}
    tables = []
    for name in names:
        with (root / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != locked[name]['sha256']:
                raise ValueError(f'Evaluation source drift: {name}')
        tables.append(pq.read_table(root / name).to_pylist())
    validation, test = tables

    def question(row):
        if row['answer'] != chr(65 + row['answer_index']):
            raise ValueError('Answer index mismatch')
        return 'Question: ' + row['question'] + '\nOptions:\n' + '\n'.join(
            f'{chr(65+i)}. {option}' for i, option in enumerate(row['options']))

    tasks = []
    for category in sorted({r['category'] for r in test}):
        examples = [r for r in validation if r['category'] == category]
        if len(examples) != 5 or not all(r['cot_content'] for r in examples):
            raise ValueError('Expected five public validation rationales per category')
        # Keep the author's rationale verbatim; do not synthesize a solution.
        prefix = '\n\n'.join(question(r) + '\n' + r['cot_content'] for r in examples)
        rows = [(prefix + '\n\n' + question(r) + "\nA: Let's think step by step.",
                 r['answer']) for r in test if r['category'] == category]
        tasks.append(Task('mmlu-pro-' + category.replace(' ', '_'), 5, True, 'choice', rows))
    yield from tasks
    # This is the complete public "ok" cohort, not all annotated Redux rows.
    # Its use follows the locked lm-eval template, not an asserted Qwen version.
    for path in sorted(ROOT.glob('redux2-ok-*.arrow')):
        subject = path.name.removeprefix('redux2-ok-').split('-data-')[0]
        with verified(path.name).open('rb') as stream:
            test = ipc.open_stream(stream).read_all().to_pylist()
        dev = pq.read_table(verified(f'mmlu-{subject}-dev.parquet')).to_pylist()

        def question(row):
            return row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(row['choices']))+'\nAnswer:'

        if len(dev) != 5 or any(r['error_type'] != 'ok' for r in test):
            raise ValueError('Unexpected Redux2 cohort or MMLU dev split')
        prefix = '\n\n'.join(question(r)+' '+chr(65+r['answer']) for r in dev)
        yield Task('redux2-ok-'+subject, 5, False, 'choice',
                   [(prefix+'\n\n'+question(r), chr(65+r['answer'])) for r in test])
    # Four authored training solutions per subject, no test-answer exemplars.
    for train_path in sorted(ROOT.glob('math-*-train-*.parquet')):
        subject = train_path.name.removeprefix('math-').split('-train-')[0]
        train = pq.read_table(verified(train_path.name)).to_pylist()
        test = pq.read_table(verified(train_path.name.replace('-train-', '-test-'))).to_pylist()
        prefix = '\n\n'.join('Problem: '+r['problem']+'\nSolution: '+r['solution'] for r in train[:4])
        rows = []
        for row in test:
            gold = boxed(row['solution'])
            if gold is None:
                raise ValueError('MATH solution has no balanced final box')
            rows.append((prefix+'\n\nProblem: '+row['problem']+"\nSolution: Let's think step by step.", gold))
        yield Task('math-'+subject, 4, True, 'boxed_math', rows)

    for path in sorted(ROOT.glob('include-*-test-*.parquet')):
        language = path.name.removeprefix('include-').split('-test-')[0]
        test, examples, _ = include_examples(path)

        def question(row):
            if row['answer'] not in range(4):
                raise ValueError('INCLUDE expects zero-based choice indices')
            return row['question']+'\n'+'\n'.join(f'{letter}. {row["option_"+letter.lower()]}' for letter in 'ABCD')+'\nAnswer:'

        prefix = '\n\n'.join(question(r)+' '+chr(65+r['answer']) for r in examples)
        yield Task('include-'+language.lower().replace(' ', '_'), 5, False, 'choice',
                   [(prefix+'\n\n'+question(r), chr(65+r['answer'])) for r in test])

    prompt_path = Path('.cache/references/supergpqa-five-shot.yaml')
    spec = next(s for s in json.loads(Path('sources.lock.json').read_text()) if s['name'] == prompt_path.name)
    raw = prompt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('SuperGPQA prompt drift')
    template = yaml.safe_load(raw)['prompt_format'][0]
    if template.count('Answer: Let') != 6:
        raise ValueError('Expected five public solutions and one query cue')
    rows = [json.loads(s) for s in verified('supergpqa-SuperGPQA-all.jsonl').read_text().splitlines()]
    prompts = []
    for row in rows:
        question = row['question']+'\n'+'\n'.join(f'{chr(65+i)}) {option}' for i, option in enumerate(row['options']))
        prompts.append((template.format(question), row['answer_letter']))
    yield Task('supergpqa', 5, True, 'choice', prompts)
