"""Bind all416 Mac RULER cases and extract exact author scoring/postprocessing."""
import ast,json,re
from pathlib import Path
from tokenizers import Tokenizer
from ruler_tasks import configurations,digest,metrics


def postprocessor():
    source=Path('.cache/ruler-sources/evaluate.py')
    spec=json.loads(Path('ruler_eval_sources.lock.json').read_text())[0];assert digest(source)==spec['sha256']
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
    namespace={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['postprocess_pred']


def cohort():
    root=Path('data/ruler-mac');meta=json.loads((root/'meta.json').read_text());assert meta['status']=='complete_frozen_mac_cohort' and meta['total_rows']==416 and len(meta['cases'])==26
    assert all(digest(name)==sha for name,sha in meta['identity']['sources'].items()),'Preparation source drift'
    assert all(digest(name)==sha for name,sha in meta['identity']['raw'].items())
    assert digest('data/ruler-source/PaulGrahamEssays.json')==meta['identity']['essay_sha256']
    configs=configurations();codec=Tokenizer.from_file('.cache/references/tokenizer.json');rows=[]
    for length in (4096,8192):
        for task,config in configs.items():
            case=meta['cases'][f'{length}/{task}'];assert digest(case['file'])==case['sha256'] and case['rows']==16
            source=[json.loads(line) for line in Path(case['file']).read_text().splitlines()];assert len(source)==16
            original=next(s for s in json.loads(Path('ruler_sources.lock.json').read_text()) if s['url'].endswith('/'+config['task']+'.py'))
            proof=json.loads((Path(case['file']).parents[1]/'child.json').read_text());assert proof['source_sha256']==original['sha256']
            command=proof['command'];assert command[command.index('--template')+1]==config['template'] and command[command.index('--random_seed')+1]=='42'
            assert command[command.index('--max_seq_length')+1]==str(length) and command[command.index('--num_samples')+1]=='16'
            for index,row in enumerate(source):
                assert row['index']==index and row['outputs'] and all(isinstance(x,str) and x for x in row['outputs'])
                prompt=row['input']+row['answer_prefix'];ids=codec.encode(prompt,add_special_tokens=False).ids
                assert len(ids)+config['tokens_to_generate']==row['length']<=length
                rows.append(dict(source_id=f'{length}/{task}:{index}',task=task,kind=config['task'],context=length,tokens=config['tokens_to_generate'],author_row=row,source_file_sha256=case['sha256'],prompt=prompt,input_ids=ids))
    assert len(rows)==416 and len({r['source_id'] for r in rows})==416
    return rows,meta


def scorer():
    process=postprocessor();functions=metrics()
    def score(response,row):return functions[row['kind']]['metric_fn']([process(response,{})],[row['author_row']['outputs']])
    return score


def summary(records):
    process=postprocessor();functions=metrics();scores={}
    for length in (4096,8192):
        for task,config in configurations().items():
            selected=[r for r in records if r['context']==length and r['task']==task];assert len(selected)==16
            scores[f'{length}/{task}']=functions[config['task']]['metric_fn']([process(r['response'],{}) for r in selected],[r['author_row']['outputs'] for r in selected])
    return dict(task_scores_percent=scores,configuration_macro_percent={str(length):sum(value for key,value in scores.items() if key.startswith(str(length)+'/'))/13 for length in (4096,8192)})
