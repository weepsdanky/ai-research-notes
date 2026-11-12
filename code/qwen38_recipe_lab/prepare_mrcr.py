"""Freeze the complete Mac-fit cohort before inference; never truncate prompts."""
import argparse,gc,hashlib,json,os,resource,signal,subprocess,time
from datetime import datetime,timedelta
from pathlib import Path
from mrcr_tasks import render,validate_source_row,locked_text
from token_capacity import CapacityBound


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:parser.error('Aware deadline required')
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['RAYON_NUM_THREADS']='1'
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    spec=json.loads(Path('profiles/mrcr_mac.json').read_text());root=Path('data/mrcr-mac');root.mkdir(parents=True,exist_ok=True)
    output=root/'prepared.jsonl';part=root/'prepared.jsonl.part';meta_path=root/'meta.json';progress=root/'progress.json'
    if output.exists() or meta_path.exists():raise FileExistsError('Preserve completed prepared cohort')
    files=['prepare_mrcr.py','token_capacity.py','mrcr_tasks.py','profiles/mrcr_mac.json','long_context_sources.lock.json','.cache/references/tokenizer.json']
    identity=dict(sources={name:digest(name) for name in files},shards=spec['shards'])
    for row in spec['shards']:
        path=Path('data/mrcr-source')/Path(row['file']).name
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Raw MRCR source drift'
    codec=Tokenizer.from_file('.cache/references/tokenizer.json');capacity=CapacityBound.load()
    tokenizer_spec=next(s for s in json.loads(Path('sources.lock.json').read_text()) if s['name']=='tokenizer.json')
    assert identity['sources']['.cache/references/tokenizer.json']==tokenizer_spec['sha256']
    # With no user literal control tokens, direct BPE matches the released template;
    # otherwise stop for a reviewed sanitizer instead of silently altering examples.
    added=json.loads(Path('.cache/references/tokenizer.json').read_text())['added_tokens']
    import re
    template_source=locked_text('qwen-chat-template.jinja')
    controls=[t['content'] for t in added if t['special'] or t['content'] in template_source]
    pattern=re.compile('|'.join(re.escape(x) for x in sorted(controls,key=len,reverse=True)))
    state=dict(identity=identity,status='running',started=datetime.now().astimezone().isoformat(),processed=0,selected=0,excluded=[],source_answer_and_eight_needles_verified=0,prepared_bytes=0,prepared_prefix_sha256=hashlib.sha256(b'').hexdigest(),max_payload_bytes_per_token=capacity.maximum,capacity_proof='NFC per pre-normalization AddedToken span; no stripped input; full byte alphabet; at most max_payload_bytes_per_token per ID')
    if args.resume:
        state=json.loads(progress.read_text());assert state['identity']==identity and part.exists()
        data=part.read_bytes();committed=data[:state['prepared_bytes']]
        assert len(committed)==state['prepared_bytes'] and hashlib.sha256(committed).hexdigest()==state['prepared_prefix_sha256'],'Prepared prefix changed'
        assert len(committed.split(b'\n'))-1==state['selected']
        if len(data)>len(committed):
            (root/f'prepared-uncommitted-{time.time_ns()}.jsonl').write_bytes(data[len(committed):])
            with part.open('r+b') as stream:stream.truncate(len(committed))
    else:
        if progress.exists() or part.exists():raise FileExistsError('Use explicit --resume for partial preparation')
        part.touch(exist_ok=False)
    tick=time.monotonic();requested=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:requested.append(signum))
    prefix_hash=hashlib.sha256(part.read_bytes())
    def save():
        tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(progress)
    def guard():
        level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;elapsed=time.monotonic()-tick
        if requested or level>=2 or peak>spec['preparation_peak_rss_cap_bytes'] or elapsed>=1180 or datetime.now().astimezone()>=deadline-timedelta(seconds=180):
            state['stop_observation']=dict(at=datetime.now().astimezone().isoformat(),pressure=level,peak_rss_bytes=peak,elapsed_seconds=elapsed,signals=list(requested))
            raise InterruptedError('Preserve partial cohort: signal/resource/time budget')
    save()
    try:
        ordinal=0
        with part.open('a') as stream:
            for shard in spec['shards']:
                file=Path('data/mrcr-source')/Path(shard['file']).name
                local_index=0
                reader=pq.ParquetFile(file)
                for batch in reader.iter_batches(batch_size=1,use_threads=False):
                    for row in batch.to_pylist():
                        source_id=f"{Path(shard['file']).name}:{local_index}"
                        ordinal+=1;local_index+=1
                        if ordinal<=state['processed']:continue
                        guard();messages=json.loads(row['prompt']);validate_source_row(row,messages)
                        assert not any(pattern.search(m['content']) for m in messages),'Literal chat control tokens require exact sanitization review'
                        text=render(messages);lower,payload_bytes=capacity.measure(text,codec.normalizer)
                        encoded=None;ids=None
                        if lower+spec['tokens']<=spec['context']:
                            encoded=codec.encode(text,add_special_tokens=False);ids=encoded.ids
                            assert lower<=len(ids),'Capacity lower bound invalid'
                        guard()
                        if ids is not None and len(ids)+spec['tokens']<=spec['context']:
                            record=dict(source_id=source_id,source_ordinal=ordinal-1,prompt=text,input_ids=ids,answer=row['answer'],prefix=row['random_string_to_prepend'],source_prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),prompt_tokens=len(ids))
                            line=json.dumps(record,ensure_ascii=False)+'\n'
                            assert stream.tell()+len(line.encode())<=spec['prepared_cap_bytes'],'Prepared data cap'
                            stream.write(line);stream.flush();prefix_hash.update(line.encode());state['selected']+=1
                            state.update(prepared_bytes=stream.tell(),prepared_prefix_sha256=prefix_hash.hexdigest())
                            del record,line
                        else:
                            proof=dict(source_id=source_id,source_ordinal=ordinal-1,reason='complete prompt plus generation exceeds context')
                            if ids is None:proof.update(prompt_token_lower_bound=lower,normalized_payload_bytes=payload_bytes,proof='byte_bound')
                            else:proof.update(prompt_tokens=len(ids),proof='full_tokenization')
                            state['excluded'].append(proof)
                        state['processed']=ordinal;state['source_answer_and_eight_needles_verified']+=1;save()
                        if ordinal%50==0:print('prepared source rows',ordinal,'eligible',state['selected'],flush=True)
                        del encoded,ids,text,messages,row
                    del batch
                reader.close();del reader
                gc.collect();pa.default_memory_pool().release_unused()
        assert ordinal==state['processed']==spec['expected_source_rows']==800
        assert state['selected']>0 and state['selected']+len(state['excluded'])==800
        state.update(status='complete_frozen_mac_cohort',ended=datetime.now().astimezone().isoformat(),prepared_sha256=digest(part),prepared_bytes=part.stat().st_size,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_prompt_truncation=True,full_paper_protocol=False)
        part.rename(output);meta_path.write_text(json.dumps(state,indent=2)+'\n');save()
        print('Frozen cohort:',state['selected'],'of800; no model inference',flush=True)
    except BaseException as error:
        state.update(status='stopped_or_error',error_type=type(error).__name__,ended=datetime.now().astimezone().isoformat());save();raise


if __name__=='__main__':main()
