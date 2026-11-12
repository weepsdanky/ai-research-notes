"""Independent streaming checks of compressed author500 corpus and real output prefixes."""
import argparse,ast,gzip,hashlib,json,resource,subprocess,time
from datetime import datetime,timedelta
from pathlib import Path
from tokenizers import Tokenizer
from ruler_tasks import configurations,digest,metrics
from ruler_evaluation import postprocessor
from ruler_public500_repair import frozen,rows,ROOT,PROFILE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',nargs='*');p.add_argument('--out',required=True);p.add_argument('--deadline',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();target=Path(a.out);assert not target.exists();deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo is not None;tick=time.monotonic()
    source_lock=json.loads(Path('sources.lock.json').read_text())
    for filename in ('tokenizer.json','generation_config.json'):assert digest(Path('.cache/references')/filename)==next(item['sha256'] for item in source_lock if item['name']==filename)
    prepared=frozen();profile=json.loads(PROFILE.read_text());previous=json.loads(Path('data/ruler-mac/meta.json').read_text());configs=configurations();codec=Tokenizer.from_file('.cache/references/tokenizer.json');functions=metrics();process=postprocessor()
    for key,case in prepared['cases'].items():
        old=previous['cases'][key];assert digest(old['file'])==old['sha256'];prefix=Path(old['file']).read_bytes()
        with gzip.open(case['file'],'rb') as stream:assert stream.read(len(prefix))==prefix
        proof=json.loads(Path(case['author_proof']).read_text());original=json.loads((Path(old['file']).parents[1]/'child.json').read_text());assert proof['source_sha256']==original['source_sha256']==digest(proof['command'][0]);wanted=list(original['command']);wanted[wanted.index('--num_samples')+1]='500';wanted[wanted.index('--save_dir')+1]=proof['command'][proof['command'].index('--save_dir')+1];assert proof['command']==wanted and proof['peak_rss_bytes']<=2000000000
        if key.endswith('/qa_2'):
            assert proof['repair']=='used_docs=max(incremental,used_docs-incremental)'
            raw_source=Path(proof['command'][0]).read_text();assert raw_source.count('used_docs -= incremental')==1
            repaired=raw_source.replace('used_docs -= incremental','used_docs = max(incremental, used_docs - incremental)')
            assert hashlib.sha256(ast.unparse(ast.parse(repaired)).encode()).hexdigest()==proof['effective_ast_sha256'],'Effective repair differs from the declared single clamp'
        else:assert proof.get('repair') is None
    total=0
    for row in rows(prepared,codec):
        total+=1
        if total%500==0:
            assert datetime.now().astimezone()<deadline-timedelta(seconds=180) and time.monotonic()-tick<1180
            assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=2000000000
    assert total==13000 and prepared['compressed_total_bytes']<=64000000;assert digest(profile['checkpoint'])==profile['checkpoint_sha256'];eos=set(json.loads(Path('.cache/references/generation_config.json').read_text())['eos_token_id']);files={}
    for filename in a.predictions:
        meta=json.loads(Path(filename+'.meta.json').read_text());identity=meta['identity'];assert identity['suite']=='ruler_public500_qa_repair' and identity['checkpoint_sha256']==profile['checkpoint_sha256'] and identity['prepared_meta_sha256']==digest(ROOT/'meta.json')
        assert identity['contexts']==[4096,8192] and identity['samples_per_configuration_per_length']==500 and identity['target_items']==13000 and identity['generation_budgets']==profile['generation_budgets'] and identity['device']=='cpu' and identity['threads']==1 and identity['batch_size']==1 and identity['greedy'] and identity['streaming']
        assert identity['stage'] in ('sparse','dense') and identity['grouped_gqa']==(identity['stage']=='sparse') and identity['dense_shared_kv']==(identity['stage']=='dense')
        assert all(digest(name)==sha for name,sha in identity['sources'].items()) and digest(filename)==meta['predictions_sha256']
        assert meta['status'] in ('complete','time_budget','interrupted_signal','memory_pressure')
        if not a.allow_partial:assert meta['status']=='complete' and meta['full_mac_cohort_generated']
        iterator=iter(rows(prepared,codec));count=0;groups={}
        with Path(filename).open() as stream:
            for line in stream:
                record=json.loads(line);row=next(iterator);assert {key:record[key] for key in row}==row;ids=record['output_ids'];assert 0<len(ids)<=row['tokens'] and not any(i in eos for i in ids[:-1])
                assert record['raw']==codec.decode(ids,skip_special_tokens=False) and record['response']==codec.decode(ids[:-1] if ids[-1] in eos else ids,skip_special_tokens=False)
                assert (record['stop']=='eos' and ids[-1] in eos) or (record['stop']=='max_new_tokens' and len(ids)==row['tokens'] and ids[-1] not in eos)
                assert record['unused_token_ids']==[i for i in ids if codec.id_to_token(i) is None]
                response=process(record['response'],{});references=row['author_row']['outputs'];assert record['score']==functions[row['kind']]['metric_fn']([response],[references]);groups.setdefault(f"{row['context']}/{row['task']}",[]).append((response,references));count+=1
        assert count==meta['completed_items']<=13000 and meta['target_items']==13000
        summary=None
        if meta['status']=='complete':
            assert count==13000 and len(groups)==26;scores={}
            for key,items in groups.items():
                assert len(items)==500;task=key.split('/')[1];scores[key]=functions[configs[task]['task']]['metric_fn']([x[0] for x in items],[x[1] for x in items])
            summary=dict(task_scores_percent=scores,configuration_macro_percent={str(length):sum(value for key,value in scores.items() if key.startswith(str(length)+'/'))/13 for length in (4096,8192)})
        assert meta['author_summary']==summary
        files[filename]=dict(items=count,target_items=13000,complete_mac_cohort=meta['status']=='complete',author_summary=summary,sha256=digest(filename),meta_sha256=digest(filename+'.meta.json'),stage=identity['stage'],all_source_prompts_gold_input_ids_output_ids_raw_response_and_scores_verified=True)
    result=dict(at=datetime.now().astimezone().isoformat(),auditor_sha256=digest(__file__),prepared_meta_sha256=digest(ROOT/'meta.json'),source_cases=13000,variant='explicit Mac QA retry-clamp repair, not unchanged author500',samples_per_configuration_per_length=500,configurations=13,lengths=[4096,8192],all13000_source_token_budgets_and_first16_byte_prefixes_verified=True,compressed_total_bytes=prepared['compressed_total_bytes'],uncompressed_total_bytes=prepared['uncompressed_total_bytes'],full_paper_protocol=False,files=files,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
