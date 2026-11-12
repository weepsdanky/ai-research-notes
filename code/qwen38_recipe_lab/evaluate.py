"""Bounded generation pilot, NOT the paper's complete evaluation suite.

GSM8K uses four fixed training examples; MGSM uses the authors' eight CoT
examples; MMLU uses five dev examples; BBH uses three author CoT examples.
Only one MMLU subject and two BBH tasks are implemented, explicitly a pilot.
Full prompts are retained, never silently truncated. Context beyond
the 128-token training length is explicitly recorded as extrapolation.
"""
import argparse, ast, csv, hashlib, json, re, time
from decimal import Decimal
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Config, Model
from train import device_for


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_tasks():
    root = Path('data/eval-source')
    for spec in json.loads(Path('eval_sources.lock.json').read_text()):
        if digest(root / spec['name']) != spec['sha256']:
            raise ValueError(f"Evaluation source hash mismatch: {spec['name']}")
    train = [json.loads(line) for line in (root/'gsm8k-train.jsonl').read_text().splitlines()]
    prefix = '\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in train[:4])
    test = [json.loads(line) for line in (root/'gsm8k-test.jsonl').read_text().splitlines()]
    tasks = {'gsm8k': [(prefix+'\n\nQuestion: '+r['question']+'\nAnswer: Let\'s think step by step.', r['answer'].split('####')[-1].strip()) for r in test]}
    # Literal-only parsing: downloaded Python is data, never executable code.
    tree = ast.parse((root/'mgsm-exemplars.py').read_text())
    examples = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'MGSM_EXEMPLARS' for t in n.targets))
    for lang in ('en', 'zh'):
        prefix = '\n\n'.join(examples[lang][str(i)]['q']+'\n'+examples[lang][str(i)]['a'] for i in range(1,9))
        suffix = '\nAnswer: Let\'s think step by step.' if lang == 'en' else '\n逐步解答：'
        rows = list(csv.reader((root/f'mgsm-{lang}.tsv').open(), delimiter='\t'))
        tasks['mgsm-'+lang] = [(prefix+'\n\n'+q+suffix, answer) for q, answer in rows]
    import pyarrow.parquet as pq
    dev=pq.read_table(root/'mmlu-algebra-dev.parquet').to_pylist()
    test=pq.read_table(root/'mmlu-algebra-test.parquet').to_pylist()
    def mmlu_question(row):
        choices='\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['choices']))
        return row['question']+'\n'+choices+'\nAnswer:'
    prefix='The following are multiple choice questions (with answers) about abstract algebra.\n\n'
    assert len(dev)==5
    prefix+='\n\n'.join(mmlu_question(r)+' '+chr(65+r['answer']) for r in dev)
    tasks['mmlu-algebra']=[(prefix+'\n\n'+mmlu_question(r),chr(65+r['answer'])) for r in test]
    for task in ('boolean_expressions','date_understanding'):
        # Remove the benchmark canary/header, retain all three author examples.
        prefix=(root/f'bbh-{task}.txt').read_text().split('-----\n',1)[1].strip()
        assert prefix.count('Q: ')==3
        examples=json.loads((root/f'bbh-{task}.json').read_text())['examples']
        tasks['bbh-'+task]=[(prefix+'\n\nQ: '+r['input']+'\nA: Let\'s think step by step.',r['target']) for r in examples]
    return tasks


def extract(text):
    values = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return str(Decimal(values[-1].replace(',', ''))) if values else None


def task_answer(task, text):
    if task in ('mmlu-algebra','bbh-date_understanding'):
        matches=re.findall(r'\b([A-F])\b',text)
        return matches[-1] if matches else None
    if task=='bbh-boolean_expressions':
        matches=re.findall(r'\b(True|False)\b',text,re.IGNORECASE)
        return matches[-1].lower() if matches else None
    return extract(text)


@torch.no_grad()
def next_logits(model, ids, stage, cache):
    # Same tested model/cache path, only project the final position to vocab.
    hidden = model.embedding(ids).repeat(1,1,model.config.branches)
    for i, layer in enumerate(model.layers):
        hidden, _, _ = layer(hidden, ids, stage, cache.setdefault(str(i), {}))
    return model.head(model.readout(hidden[:, -1:]))[0, -1]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu','mps','auto'])
    p.add_argument('--limit', type=int, default=2); p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--tasks',nargs='+',help='Optional task names; default all implemented pilots')
    a = p.parse_args()
    if min(a.limit,a.tokens,a.context) < 1: p.error('Positive budgets required')
    torch.set_num_threads(1); torch.manual_seed(42)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if saved['step'] != saved['args']['steps']: raise ValueError('Use a completed phase checkpoint')
    device = device_for(a.device)
    model = Model(Config(**saved['config'])).to(device).eval(); model.load_state_dict(saved['model'])
    tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
    stage = 'sparse' if saved['args']['stage'] == 'sparse' else 'pretrain'
    metadata = dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=digest(a.checkpoint),
        checkpoint_step=saved['step'], config=saved['config'], training_context=saved['args']['seq'],
        model_sources=saved['source_hashes'], evaluator_sha256=digest(__file__),
        source_lock_sha256=digest('eval_sources.lock.json'), tokenizer_sha256=digest('.cache/references/tokenizer.json'),
        device=device, greedy=True, max_new_tokens=a.tokens, max_context=a.context,
        extraction='GSM/MGSM:last numeric, commas removed, Decimal equality; MMLU/date:last standalone uppercase A-F; Boolean:last True/False',
        status='bounded pilot; not complete official evaluation; Mac prompt/decoding choices')
    totals = {}
    with Path(a.out).open('x') as stream:
        for task, rows in load_tasks().items():
            if a.tasks and task not in a.tasks: continue
            correct = 0; failures = 0
            for i, (prompt,gold) in enumerate(rows[:a.limit]):
                ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                shots=4 if task=='gsm8k' else (5 if task=='mmlu-algebra' else (3 if task.startswith('bbh-') else 8))
                record = dict(task=task,index=i,shots=shots,cot=task!='mmlu-algebra',
                    prompt=prompt,input_ids=ids,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    gold=gold,prompt_tokens=len(ids),context_extrapolation=len(ids)>saved['args']['seq'])
                generated=[]; started=time.perf_counter(); stop='max_new_tokens'
                if len(ids)+a.tokens>a.context:
                    stop='context_capacity'; failures+=1
                else:
                    cache={}; logits=next_logits(model,torch.tensor([ids],device=device),stage,cache)
                    prefill=time.perf_counter()-started
                    for step in range(a.tokens):
                        token=int(logits.argmax().item()); generated.append(token)
                        if token==model.config.eos: stop='eos'; break
                        if step+1<a.tokens:
                            logits=next_logits(model,torch.tensor([[token]],device=device),stage,cache)
                    record['prefill_seconds']=prefill
                text=tokenizer.decode(generated,skip_special_tokens=False); answer=task_answer(task,text)
                target=task_answer(task,gold)
                numeric=task=='gsm8k' or task.startswith('mgsm-')
                scored=answer is not None and (Decimal(answer)==Decimal(target) if numeric else answer==target)
                correct+=int(scored)
                record.update(output_ids=generated,prediction=text,extracted=answer,correct=scored,
                    unused_token_ids=[t for t in generated if tokenizer.id_to_token(t) is None],
                    stop=stop,seconds=time.perf_counter()-started)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
                print(task,i,stop,scored,flush=True)
            n=min(a.limit,len(rows)); totals[task]=dict(n=n,total_dataset=len(rows),correct=correct,
                accuracy=correct/n,capacity_failures=failures)
    metadata['tasks']=totals
    Path(a.out+'.meta.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__': main()
