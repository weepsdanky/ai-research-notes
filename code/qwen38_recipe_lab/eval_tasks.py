"""Locked, complete core evaluation datasets and explicit Mac prompt choices.

These adapters preserve public shot/CoT counts, not undisclosed Qwen prompts.
BBH has 27 configurations in 23 families; report both macro and micro scores.
"""
import ast,csv,hashlib,json,re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq


@dataclass
class Task:
    name:str
    shots:int
    cot:bool
    kind:str
    rows:list


def answer(text,kind):
    if kind=='choice':
        matches=re.findall(r'\banswer\s*(?:is|:)\s*\(?([A-Z])\)?\b',text,re.IGNORECASE)
        if matches:return matches[-1].upper()
        matches=re.findall(r'\(([A-Z])\)',text)
        if matches:return matches[-1]
        match=re.match(r'^\s*([A-Z])[.)]?\s*(?:\n|$)',text)
        return match.group(1) if match else None
    if kind in ('boolean','yesno'):
        pattern=r'\b(True|False)\b' if kind=='boolean' else r'\b(Yes|No)\b'
        matches=re.findall(pattern,text,re.IGNORECASE)
        return matches[-1].lower() if matches else None
    if kind=='number':
        values=re.findall(r'-?\d[\d,]*(?:\.\d+)?',text)
        return str(Decimal(values[-1].replace(',',''))) if values else None
    # Free-form BBH answers (word sorting, Dyck strings, valid/invalid).
    matches=re.findall(r'(?:so\s+)?(?:the\s+)?answer\s*(?:is|:)\s*(.+)',text,re.IGNORECASE)
    value=matches[-1] if matches else text
    value=value.replace('<|endoftext|>','').strip().rstrip('.')
    return ' '.join(value.split()).casefold() or None


def correct(prediction,gold,kind):
    actual,target=answer(prediction,kind),answer(gold,kind)
    if actual is None or target is None:return False
    return Decimal(actual)==Decimal(target) if kind=='number' else actual==target


def load_tasks():
    root=Path('data/eval-source')
    for spec in json.loads(Path('eval_sources.lock.json').read_text()):
        with (root/spec['name']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=spec['sha256']:
                raise ValueError(f"Evaluation source drift: {spec['name']}")
    suite=json.loads(Path('eval_suites.json').read_text());tasks=[]
    train=[json.loads(s) for s in (root/'gsm8k-train.jsonl').read_text().splitlines()]
    test=[json.loads(s) for s in (root/'gsm8k-test.jsonl').read_text().splitlines()]
    prefix='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in train[:4])
    tasks.append(Task('gsm8k',4,True,'number',[(prefix+'\n\nQuestion: '+r['question']+"\nAnswer: Let's think step by step.",r['answer'].split('####')[-1].strip()) for r in test]))
    tree=ast.parse((root/'mgsm-exemplars.py').read_text())
    examples=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MGSM_EXEMPLARS' for t in n.targets))
    def label(text):
        match=re.search('[:：]',text)
        if match is None:raise ValueError('Missing public exemplar label')
        return text[:match.end()]+' '
    for lang in suite['mgsm_languages']:
        prefix='\n\n'.join(examples[lang][str(i)]['q']+'\n'+examples[lang][str(i)]['a'] for i in range(1,9))
        with (root/f'mgsm-{lang}.tsv').open() as stream:rows=list(csv.reader(stream,delimiter='\t'))
        q_label,a_label=label(examples[lang]['1']['q']),label(examples[lang]['1']['a'])
        tasks.append(Task('mgsm-'+lang,8,True,'number',[(prefix+'\n\n'+q_label+q+'\n'+a_label,gold) for q,gold in rows]))
    def question(row):
        return row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['choices']))+'\nAnswer:'
    for subject in suite['mmlu_subjects']:
        dev=pq.read_table(root/f'mmlu-{subject}-dev.parquet').to_pylist()
        test=pq.read_table(root/f'mmlu-{subject}-test.parquet').to_pylist();assert len(dev)==5
        prefix=f"The following are multiple choice questions (with answers) about {subject.replace('_',' ')}.\n\n"
        prefix+='\n\n'.join(question(r)+' '+chr(65+r['answer']) for r in dev)
        tasks.append(Task('mmlu-'+subject,5,False,'choice',[(prefix+'\n\n'+question(r),chr(65+r['answer'])) for r in test]))
    for name in suite['bbh_subtasks']:
        prefix=(root/f'bbh-{name}.txt').read_text().split('-----\n',1)[1].strip()
        assert prefix.count('Q: ')==3,name
        rows=json.loads((root/f'bbh-{name}.json').read_text())['examples']
        targets={r['target'] for r in rows}
        if all(re.fullmatch(r'\([A-Z]\)',s) for s in targets):kind='choice'
        elif targets<={'True','False'}:kind='boolean'
        elif targets<={'Yes','No'}:kind='yesno'
        elif all(re.fullmatch(r'-?\d+(?:\.\d+)?',s) for s in targets):kind='number'
        else:kind='text'
        tasks.append(Task('bbh-'+name,3,True,kind,[(prefix+'\n\nQ: '+r['input']+"\nA: Let's think step by step.",r['target']) for r in rows]))
    return tasks
