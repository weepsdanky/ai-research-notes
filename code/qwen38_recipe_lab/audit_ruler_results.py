"""Audit every raw RULER manifest, prompt, token and exact author score independently."""
import argparse,hashlib,json,resource,subprocess,time
from datetime import datetime,timedelta
from pathlib import Path
from tokenizers import Tokenizer
from ruler_tasks import configurations,digest,metrics
from ruler_evaluation import postprocessor


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('predictions',nargs='*');parser.add_argument('--out',required=True);parser.add_argument('--deadline',required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    target=Path(args.out);assert not target.exists();deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None;tick=time.monotonic()
    prepared=json.loads(Path('data/ruler-mac/meta.json').read_text());assert prepared['status']=='complete_frozen_mac_cohort' and prepared['total_rows']==416 and len(prepared['cases'])==26
    for name,sha in prepared['identity']['sources'].items():assert digest(name)==sha
    for item in json.loads(Path('ruler_data.lock.json').read_text())['files']:assert digest(item['file'])==item['sha256']
    conversion=json.loads(Path('runs/ruler-essay-conversion.json').read_text());assert len(conversion['inputs'])==218 and digest(conversion['file'])==conversion['sha256']==prepared['identity']['essay_sha256']
    assert digest('.cache/long-context-sources/ruler-scripts-data-synthetic-json-download_paulgraham_essay.py')==conversion['author_source_sha256']
    for item in conversion['inputs']:assert digest(item['file'])==item['sha256']
    configurations_map=configurations();codec=Tokenizer.from_file('.cache/references/tokenizer.json');functions=metrics();process=postprocessor();rows=[];author_sources=json.loads(Path('ruler_sources.lock.json').read_text())
    for length in (4096,8192):
        for task,config in configurations_map.items():
            assert datetime.now().astimezone()<deadline-timedelta(seconds=180) and time.monotonic()-tick<1180
            assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=2000000000
            case=prepared['cases'][f'{length}/{task}'];assert digest(case['file'])==case['sha256'] and case['rows']==16
            path=Path(case['file']);source=[json.loads(line) for line in path.read_text().splitlines()];assert len(source)==16
            proof=json.loads((path.parents[1]/'child.json').read_text());spec=next(s for s in author_sources if s['url'].endswith('/'+config['task']+'.py'))
            assert proof['source_sha256']==spec['sha256'] and proof['peak_rss_bytes']<=2000000000
            command=proof['command'];options={'--template':config['template'],'--random_seed':'42','--max_seq_length':str(length),'--num_samples':'16','--tokens_to_generate':str(config['tokens_to_generate']),'--subset':'validation','--tokenizer_type':'hf','--save_name':task,**{'--'+k:str(v) for k,v in config['args'].items()}}
            for key,value in options.items():assert command[command.index(key)+1]==value
            assert '--remove_newline_tab' not in command
            for index,row in enumerate(source):
                assert row['index']==index and row['outputs'] and all(isinstance(x,str) and x for x in row['outputs'])
                prompt=row['input']+row['answer_prefix'];ids=codec.encode(prompt,add_special_tokens=False).ids
                assert len(ids)+config['tokens_to_generate']==row['length']<=length
                if config['task']=='niah':assert all(answer in prompt for answer in row['outputs'])
                rows.append(dict(source_id=f'{length}/{task}:{index}',task=task,kind=config['task'],context=length,tokens=config['tokens_to_generate'],author_row=row,source_file_sha256=case['sha256'],prompt=prompt,input_ids=ids))
    assert len(rows)==416 and len({r['source_id'] for r in rows})==416
    eos=set(json.loads(Path('.cache/references/generation_config.json').read_text())['eos_token_id']);files={};profile=json.loads(Path('profiles/ruler_mac.json').read_text());assert digest(profile['checkpoint'])==profile['checkpoint_sha256']
    for filename in args.predictions:
        meta=json.loads(Path(filename+'.meta.json').read_text());identity=meta['identity'];assert identity['suite']=='ruler' and identity['checkpoint_sha256']==profile['checkpoint_sha256']
        assert identity['prepared_meta_sha256']==digest('data/ruler-mac/meta.json') and identity['generation_budgets']==profile['generation_budgets'] and identity['contexts']==[4096,8192]
        assert identity['device']=='cpu' and identity['threads']==1 and identity['greedy'] and identity['batch_size']==1 and identity['prefill_chunk']==128 and identity['gdn_scan']=='chunk'
        assert identity['stage'] in ('sparse','dense') and identity['grouped_gqa']==(identity['stage']=='sparse')
        assert all(digest(name)==sha for name,sha in identity['sources'].items())
        assert meta['status'] in ('complete','time_budget','interrupted_signal','memory_pressure')
        if not args.allow_partial:assert meta['status']=='complete' and meta['full_mac_cohort_generated']
        records=[json.loads(line) for line in Path(filename).read_text().splitlines()];assert len(records)==meta['completed_items']<=416 and meta['target_items']==416 and digest(filename)==meta['predictions_sha256']
        for record,row in zip(records,rows,strict=False):
            assert {key:record[key] for key in row}==row
            ids=record['output_ids'];assert 0<len(ids)<=row['tokens'] and not any(i in eos for i in ids[:-1])
            assert record['raw']==codec.decode(ids,skip_special_tokens=False)
            assert (record['stop']=='eos' and ids[-1] in eos) or (record['stop']=='max_new_tokens' and len(ids)==row['tokens'] and ids[-1] not in eos)
            assert record['response']==codec.decode(ids[:-1] if ids[-1] in eos else ids,skip_special_tokens=False)
            assert record['score']==functions[row['kind']]['metric_fn']([process(record['response'],{})],[row['author_row']['outputs']])
            assert record['unused_token_ids']==[i for i in ids if codec.id_to_token(i) is None]
        summary=None
        if meta['status']=='complete':
            assert len(records)==416;scores={}
            for length in (4096,8192):
                for task,config in configurations_map.items():
                    selected=[r for r in records if r['context']==length and r['task']==task];assert len(selected)==16
                    scores[f'{length}/{task}']=functions[config['task']]['metric_fn']([process(r['response'],{}) for r in selected],[r['author_row']['outputs'] for r in selected])
            summary=dict(task_scores_percent=scores,configuration_macro_percent={str(length):sum(value for key,value in scores.items() if key.startswith(str(length)+'/'))/13 for length in (4096,8192)})
        assert meta['author_summary']==summary
        files[filename]=dict(sha256=digest(filename),meta_sha256=digest(filename+'.meta.json'),stage=identity['stage'],items=len(records),complete_mac_cohort=meta['status']=='complete',author_summary=summary,all_source_prompts_gold_input_ids_output_ids_raw_response_and_scores_verified=True)
    result=dict(at=datetime.now().astimezone().isoformat(),auditor_sha256=digest(__file__),prepared_meta_sha256=digest('data/ruler-mac/meta.json'),checkpoint_sha256=profile['checkpoint_sha256'],source_cases=416,configurations=13,lengths=[4096,8192],samples_per_configuration_per_length=16,raw218_essays_and_core3_hashes_verified=True,full_paper_protocol=False,scope='Independent locked manifest/author command/template/all416 token budgets; exact author postprocessing and metrics on optional real outputs; not full-length/full-sample paper RULER',files=files,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
