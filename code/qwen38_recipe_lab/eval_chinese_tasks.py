"""Locked C-Eval labeled validation and CMMLU test, five-shot answer-only."""
import ast
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq

from eval_tasks import Task
from qwen35_quality_data import digest, load


LETTERS = 'ABCD'


def correct(prediction, gold, kind):
    assert kind == 'choice' and gold in LETTERS and len(gold) == 1
    return prediction.strip() == gold


def checked_files(suite):
    files = {}
    for item in load('chinese_eval_sources.lock.json')['files']:
        if item['suite'] != suite:
            continue
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        files[item['name']] = path
    return files


def question(row, *, cmmlu, exemplar):
    stem = row['Question' if cmmlu else 'question']
    answer = row['Answer' if cmmlu else 'answer']
    assert isinstance(stem, str) and stem.strip() and len(answer) == 1 and answer in LETTERS
    options = [f'{letter}. {row[letter]}' for letter in LETTERS]
    if cmmlu:
        text = '\n'.join(['题目：'+stem,*options,'答案是：'])
    else:
        text = '\n'.join([stem,*options,'答案：'])
    return text+(answer if exemplar else ''), answer


def build_task(name, subjects):
    rows, groups = [], []
    cmmlu = name == 'cmmlu'
    for subject, label, dev, test in subjects:
        assert len(dev) == 5 and test
        intro = (f'以下是关于{label}的单项选择题，请直接给出正确答案的选项。' if cmmlu
                 else f'以下是中国关于{label}考试的单项选择题，请选出其中的正确答案。')
        examples = [question(row,cmmlu=cmmlu,exemplar=True)[0] for row in dev]
        prefix = '\n\n'.join([intro,*examples])+'\n\n'
        for row in test:
            prompt, answer = question(row,cmmlu=cmmlu,exemplar=False)
            rows.append((prefix+prompt,answer))
            groups.append(subject)
    task = Task(name,5,False,'choice',rows)
    task.prefix_groups = groups
    return task


def load_named(suite):
    files = checked_files(suite)
    if suite == 'ceval':
        mapping = load(files['subject_mapping.json'])
        assert len(mapping) == 52
        sources = []
        for subject, (_, label, _) in sorted(mapping.items()):
            dev = pq.read_table(files[f'{subject}/dev-00000-of-00001.parquet']).to_pylist()
            val = pq.read_table(files[f'{subject}/val-00000-of-00001.parquet']).to_pylist()
            sources.append((subject,label,dev,val))
        task = build_task(suite,sources)
        assert len(task.rows)==1346
        return task
    if suite == 'cmmlu':
        tree = ast.parse(files['src/categories.py'].read_text())
        entry = next(node for node in tree.body if isinstance(node,ast.Assign)
                     and any(isinstance(target,ast.Name) and target.id=='name_en2zh' for target in node.targets))
        mapping = ast.literal_eval(entry.value)
        assert len(mapping)==67
        sources=[]
        with ZipFile(files['cmmlu_v1_0_1.zip']) as archive:
            csv_names = {name for name in archive.namelist() if name.endswith('.csv')}
            assert csv_names == {f'{split}/{subject}.csv' for split in ('dev','test') for subject in mapping}
            for subject,label in sorted(mapping.items()):
                rows=[]
                for split in ('dev','test'):
                    name=f'{split}/{subject}.csv'
                    assert archive.getinfo(name).file_size < 1_000_000
                    with archive.open(name) as stream:
                        text=io.TextIOWrapper(stream,encoding='utf-8-sig',newline='')
                        rows.append(list(csv.DictReader(text)))
                sources.append((subject,label,*rows))
        task=build_task(suite,sources)
        assert len(task.rows)==11582
        return task
    raise ValueError('Unknown Chinese benchmark')


def load_tasks():
    yield load_named('ceval')
    yield load_named('cmmlu')
