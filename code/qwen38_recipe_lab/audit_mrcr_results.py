"""Independently bind MRCR outputs to original sources, token IDs and author metric."""
import argparse,gc,hashlib,json,resource,subprocess,time
from datetime import datetime,timedelta
from pathlib import Path
from mrcr_tasks import render,author_grade
from token_capacity import CapacityBound


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_record(record,row,codec,tokens,eos,grade):
    assert {key:record[key] for key in row}==row,'Source row changed'
    ids=record['output_ids'];assert 0<len(ids)<=tokens and not any(i in eos for i in ids[:-1]),'Invalid output extent'
    assert record['raw']==codec.decode(ids,skip_special_tokens=False),'Raw decoding mismatch'
    assert (record['stop']=='eos' and ids[-1] in eos) or (record['stop']=='max_new_tokens' and len(ids)==tokens and ids[-1] not in eos),'Incomplete row scored'
    content=ids[:-1] if ids[-1] in eos else ids
    assert record['response']==codec.decode(content,skip_special_tokens=False),'Response altered'
    score=grade(record['response'],row['answer'],row['prefix'])
    assert record['score']==score and 0<=score<=1,'Author score mismatch'
    assert record['unused_token_ids']==[i for i in ids if codec.id_to_token(i) is None]
    return score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',nargs='*');parser.add_argument('--out',required=True)
    parser.add_argument('--deadline',required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();target=Path(args.out)
    if target.exists():raise FileExistsError('Preserve existing audit')
    deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None
    tick=time.monotonic()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    spec=json.loads(Path('profiles/mrcr_mac.json').read_text());prepared=json.loads(Path('data/mrcr-mac/meta.json').read_text())
    assert prepared['status']=='complete_frozen_mac_cohort' and prepared['processed']==800 and prepared['no_prompt_truncation']
    assert prepared['source_answer_and_eight_needles_verified']==800
    assert digest('data/mrcr-mac/prepared.jsonl')==prepared['prepared_sha256']
    assert all(digest(name)==sha for name,sha in prepared['identity']['sources'].items()),'Prepared source drift'
    with Path('data/mrcr-mac/prepared.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    assert len(rows)==prepared['selected'] and len(rows)+len(prepared['excluded'])==800
    wanted={row['source_id']:row for row in rows};excluded={row['source_id']:row for row in prepared['excluded']}
    assert len(wanted)==len(rows) and len(excluded)==len(prepared['excluded']) and not(set(wanted)&set(excluded))
    codec=Tokenizer.from_file('.cache/references/tokenizer.json');capacity=CapacityBound.load();grade=author_grade();matched=set();ordinal=0
    for shard in spec['shards']:
        path=Path('data/mrcr-source')/Path(shard['file']).name
        assert path.stat().st_size==shard['bytes'] and digest(path)==shard['sha256']
        index=0;reader=pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=1,use_threads=False):
            for original in batch.to_pylist():
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=spec['preparation_peak_rss_cap_bytes']
                assert time.monotonic()-tick<1180 and datetime.now().astimezone()<deadline-timedelta(seconds=180)
                assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1
                name=f'{path.name}:{index}';index+=1
                messages=json.loads(original['prompt']);request=original['desired_msg_index']
                assert original['n_needles']==8 and messages[request]['role']=='user' and messages[request+1]['role']=='assistant'
                assert original['answer']==original['random_string_to_prepend']+messages[request+1]['content']
                assert sum(m['role']=='user' and m['content']==messages[request]['content'] for m in messages[:-1])==8
                text=render(messages);lower,size=capacity.measure(text,codec.normalizer)
                if name in wanted:
                    row=wanted[name]
                    assert row['source_ordinal']==ordinal and row['answer']==original['answer'] and row['prefix']==original['random_string_to_prepend']
                    assert row['source_prompt_sha256']==hashlib.sha256(original['prompt'].encode()).hexdigest()
                    assert row['prompt']==text and row['input_ids']==codec.encode(text,add_special_tokens=False).ids
                    assert row['prompt_tokens']==len(row['input_ids']) and len(row['input_ids'])+spec['tokens']<=spec['context']
                else:
                    row=excluded[name];assert row['source_ordinal']==ordinal
                    if row['proof']=='byte_bound':assert row['prompt_token_lower_bound']==lower and row['normalized_payload_bytes']==size and lower+spec['tokens']>spec['context']
                    else:
                        ids=codec.encode(text,add_special_tokens=False).ids
                        assert row['prompt_tokens']==len(ids) and len(ids)+spec['tokens']>spec['context'];del ids
                matched.add(name);ordinal+=1
                del messages,text,original,row
            del batch
        reader.close();del reader;gc.collect();pa.default_memory_pool().release_unused()
    assert ordinal==800 and matched==set(wanted)|set(excluded)
    assert digest(spec['checkpoint'])==spec['checkpoint_sha256']
    eos=set(json.loads(Path('.cache/references/generation_config.json').read_text())['eos_token_id']);files={}
    for filename in args.predictions:
        meta=json.loads(Path(filename+'.meta.json').read_text());identity=meta['identity']
        assert identity['suite']=='mrcr' and identity['checkpoint_sha256']==spec['checkpoint_sha256']
        assert identity['prepared_sha256']==prepared['prepared_sha256'] and identity['prepared_meta_sha256']==digest('data/mrcr-mac/meta.json')
        assert identity['tokens']==1024 and identity['context']==8192 and identity['device']=='cpu' and identity['threads']==1 and identity['greedy'] and identity['batch_size']==1
        assert identity['stage'] in ('sparse','dense') and identity['grouped_gqa']==(identity['stage']=='sparse')
        assert all(digest(name)==sha for name,sha in identity['sources'].items()),'Generation source drift'
        assert meta['status'] in ('complete','time_budget','interrupted_signal','memory_pressure')
        if not args.allow_partial:assert meta['status']=='complete' and meta['full_mac_cohort_generated']
        with Path(filename).open() as stream:records=[json.loads(line) for line in stream]
        assert len(records)==meta['completed_items']<=len(rows) and meta['target_items']==len(rows)
        assert digest(filename)==meta['predictions_sha256']
        scores=[verify_record(record,row,codec,spec['tokens'],eos,grade) for record,row in zip(records,rows,strict=False)]
        complete=meta['status']=='complete' and len(records)==len(rows)
        mean=sum(scores)/len(scores) if complete else None
        assert meta['mean_author_score']==mean
        files[filename]=dict(sha256=digest(filename),meta_sha256=digest(filename+'.meta.json'),stage=identity['stage'],items=len(records),target_items=len(rows),complete_mac_cohort=complete,mean_author_score=mean,all_source_prompts_gold_input_ids_output_ids_raw_response_and_scores_verified=True)
    result=dict(at=datetime.now().astimezone().isoformat(),auditor_sha256=digest(__file__),checkpoint_sha256=spec['checkpoint_sha256'],prepared_sha256=prepared['prepared_sha256'],original_source_rows=800,selected_cohort=len(rows),excluded_context=len(prepared['excluded']),files=files,all800_source_bindings_and_admission_decisions_recomputed=True,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent raw800 source/target binding, full selected tokenization, exact or proven byte-bound exclusion, and optional prediction audit',full_paper_protocol=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
