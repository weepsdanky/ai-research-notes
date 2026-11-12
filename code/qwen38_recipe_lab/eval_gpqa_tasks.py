"""Pinned GPQA author CoT and explicitly separate clean Mac templates."""
import csv
import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from eval_tasks import Task

SEED = 42
PATTERNS = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']


def normalize(text):
    return ' '.join(text.casefold().split())


@lru_cache
def sources():
    lock = json.loads(Path('gpqa_sources.lock.json').read_text())
    for spec in lock['data'] + lock['references']:
        if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('GPQA source drift: ' + spec['path'])
    data = {}
    for spec in lock['data']:
        with Path(spec['path']).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == spec['rows']
        assert len({row['Record ID'] for row in rows}) == len(rows)
        data[Path(spec['path']).stem.removeprefix('gpqa_')] = rows
    examples = json.loads(Path('.cache/references/gpqa-author/prompts__chain_of_thought_examples.json').read_text())['questions']
    assert len(examples) == 5
    return data, examples


def fewshot(mode):
    data, original = sources()
    examples = [dict(row) for row in original]
    if mode == 'author':
        return examples
    if mode != 'clean':
        raise ValueError(mode)
    tests = {normalize(row['Question']) for name in ('main', 'diamond') for row in data[name]}
    overlap = [i for i, row in enumerate(examples) if normalize(row['question']) in tests]
    assert overlap == [2, 3]
    excluded = tests | {normalize(row['question']) for row in examples}
    candidates = sorted((row for row in data['extended'] if normalize(row['Question']) not in excluded and row['Explanation'].strip()), key=lambda row: row['Record ID'])
    for index, row in zip(overlap, candidates):
        choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        assert len(set(choices)) == 4
        examples[index] = dict(question=row['Question'], choices=dict(zip('ABCD', choices)), explanation=row['Explanation'], correct_answer='D')
    assert len({normalize(row['question']) for row in examples}) == 5
    assert not any(normalize(row['question']) in tests for row in examples)
    return examples


def prompt(question, choices, examples):
    text = 'Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n'
    for row in examples:
        text += f'Question: {row["question"]}\nChoices:\n'
        for key, value in row['choices'].items():
            text += f'({key}) {value}\n'
        text += f"Let's think step by step: \n{row['explanation']}\n"
        text += f'The correct answer is ({row["correct_answer"]})\n'
    text += f'Question: {question}\nChoices:\n' + '\n'.join(f'({key}) {value}' for key, value in zip('ABCD', choices))
    return text + '\nGive step by step reasoning before you answer, and when you\'re ready to answer, please use the format "The correct answer is (insert answer here)":\n'


def load_tasks():
    data, _ = sources()
    for mode in ('author', 'clean'):
        examples = fewshot(mode)
        for name in ('main', 'diamond'):
            rng = random.Random(SEED)
            rows = []
            for row in data[name]:
                choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
                # Preserve publisher duplicate options and its first-index gold rule.
                rng.shuffle(choices)
                rows.append((prompt(row['Question'], choices, examples), 'ABCD'[choices.index(row['Correct Answer'])]))
            yield Task(f'gpqa-{mode}-{name}', 5, True, 'gpqa_choice', rows)


def extract(text):
    for pattern in PATTERNS:
        match = re.search(pattern, text)
        if match and match.group(1) in 'ABCD':
            return match.group(1)
    return None


def correct(prediction, gold, kind):
    if kind != 'gpqa_choice' or gold not in 'ABCD' or len(gold) != 1:
        raise ValueError('Invalid GPQA scorer inputs')
    return extract(prediction) == gold
