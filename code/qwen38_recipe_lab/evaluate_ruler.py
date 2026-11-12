"""Serial all416 Mac RULER cases: real greedy outputs and locked author scoring."""
import argparse,fcntl,hashlib,json,platform,resource,time
from datetime import datetime,timedelta
from pathlib import Path
import torch
from tokenizers import Tokenizer
from evaluate_batched import stop_requests,system_pressure
from evaluate_suite import save_meta,now
from ruler_evaluation import cohort,scorer,summary
from model import Config,Model
from prefill_inference import PrefillInference
from shared_kv_inference import SharedKVDenseInference


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()



@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['sparse','dense'],required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--deadline',required=True);parser.add_argument('--seconds',type=int,default=1200);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();call_started=time.monotonic();effective_seconds=min(args.seconds,1180);deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or not 1<=args.seconds<=1200:parser.error('Aware deadline and seconds1..1200 required')
    if datetime.now().astimezone()>=deadline-timedelta(seconds=200):raise ValueError('No model launch: cleanup reserve reached')
    if system_pressure()!=1:raise RuntimeError('Review current memory pressure before model launch')
    rows,prepared=cohort();spec=json.loads(Path('profiles/ruler_mac.json').read_text())
    checkpoint=spec['checkpoint'];assert digest(checkpoint)==spec['checkpoint_sha256']
    sources=['evaluate_ruler.py','shared_kv_inference.py','ruler_evaluation.py','ruler_tasks.py','ruler_sources.lock.json','ruler_eval_sources.lock.json','ruler_data.lock.json','ruler_nltk.lock.json','evaluate_batched.py','eval_tasks.py','evaluate.py','evaluate_suite.py','model.py','optim.py','polar_coefficients.json','train.py','inference.py','prefill_inference.py','reference.py','sources.lock.json','long_context_sources.lock.json','profiles/ruler_mac.json','.cache/references/modeling_qwen4_exp.py','.cache/references/tokenizer.json','.cache/references/generation_config.json']
    identity=dict(suite='ruler',checkpoint_sha256=spec['checkpoint_sha256'],prepared_meta_sha256=digest('data/ruler-mac/meta.json'),stage=args.stage,generation_budgets=spec['generation_budgets'],contexts=spec['lengths'],device='cpu',threads=1,greedy=True,batch_size=1,prefill_chunk=128,gdn_scan='chunk',grouped_gqa=args.stage=='sparse',dense_shared_kv=args.stage=='dense',sources={name:digest(name) for name in sources})
    codec=Tokenizer.from_file('.cache/references/tokenizer.json');grade=scorer()
    eos_spec=next(x for x in json.loads(Path('sources.lock.json').read_text()) if x['name']=='generation_config.json')
    assert digest('.cache/references/generation_config.json')==eos_spec['sha256']
    eos=set(json.loads(Path('.cache/references/generation_config.json').read_text())['eos_token_id'])
    for row in rows:
        assert codec.encode(row['prompt'],add_special_tokens=False).ids==row['input_ids']
        assert len(row['input_ids'])+row['tokens']<=row['context']
    if time.monotonic()-call_started>=effective_seconds:raise ValueError('No model launch: wall budget spent in preflight')
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True);meta_path=Path(str(out)+'.meta.json')
    with Path(str(out)+'.lock').open('a') as lock,stop_requests() as requested:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);existing=[]
        if args.resume:
            meta=json.loads(meta_path.read_text());assert meta['identity']==identity,'Resume identity mismatch'
            assert meta['status'] in ('time_budget','interrupted_signal','memory_pressure'),'Review terminal state before resuming'
            with out.open() as stream:existing=[json.loads(line) for line in stream]
            for row,record in zip(rows,existing,strict=False):
                assert {key:record[key] for key in row}==row,'Stored RULER source row changed'
                ids=record['output_ids'];assert 0<len(ids)<=row['tokens'] and not any(i in eos for i in ids[:-1])
                assert record['raw']==codec.decode(ids,skip_special_tokens=False)
                assert (record['stop']=='eos' and ids[-1] in eos) or (record['stop']=='max_new_tokens' and len(ids)==row['tokens'] and ids[-1] not in eos)
                content=ids[:-1] if ids[-1] in eos else ids
                assert record['response']==codec.decode(content,skip_special_tokens=False)
                assert record['score']==grade(record['response'],row)
            assert len(existing)==meta['completed_items']<=len(rows)
            if 'unscored_incomplete_row' in meta:meta.setdefault('incomplete_row_history',[]).append(meta.pop('unscored_incomplete_row'))
        else:
            if out.exists() or meta_path.exists():raise FileExistsError('Preserve existing predictions; use explicit --resume')
            out.touch(exist_ok=False);meta=dict(identity=identity,started=now(),events=[])
        torch.set_num_threads(1);torch.manual_seed(42)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        assert saved['step']==saved['args']['steps']==8000
        assert saved['source_hashes']=={name:identity['sources'][name] for name in ('model.py','optim.py','polar_coefficients.json','train.py')}
        saved.pop('optimizers',None);model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'),assign=True);del saved
        runner=PrefillInference(model,'sparse',prefill_chunk=128,grouped_gqa=True) if args.stage=='sparse' else SharedKVDenseInference(model,prefill_chunk=128)
        tick=call_started;completed=len(existing);records=list(existing)
        meta.update(status='running',completed_items=completed,target_items=len(rows),configurations=13,lengths=spec['lengths'],samples_per_configuration_per_length=16,full_paper_protocol=False,scope='All13 configurations at4K/8K and16 samples each; not the original128K-1M/full-sample protocol',environment=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform()))
        meta['events'].append(dict(start=now(),resume=args.resume,starting_items=completed,budget_seconds=effective_seconds,requested_seconds=args.seconds));save_meta(meta_path,meta)
        def stop_reason():
            if requested:return 'interrupted_signal'
            if system_pressure()>=2:return 'memory_pressure'
            if time.monotonic()-tick>=effective_seconds or datetime.now().astimezone()>=deadline-timedelta(seconds=180):return 'time_budget'
            return None
        try:
            with out.open('a') as stream:
                for row in rows[completed:]:
                    reason=stop_reason()
                    if reason:meta['status']=reason;break
                    start=time.perf_counter();cache={};last=runner(torch.tensor([row['input_ids']]),cache);prefill=time.perf_counter()-start
                    generated=[];stop='max_new_tokens';aborted=None
                    for step in range(row['tokens']):
                        if step%32==0 and (aborted:=stop_reason()):break
                        if not torch.isfinite(last).all():raise FloatingPointError('Nonfinite generation logits')
                        token=int(last.argmax(-1).item());generated.append(token)
                        if token in eos:stop='eos';break
                        if step+1<row['tokens']:last=runner(torch.tensor([[token]]),cache)
                    if aborted:
                        # This incomplete row is never scored; deterministic resume replays it.
                        meta['status']=aborted
                        meta['unscored_incomplete_row']=dict(source_id=row['source_id'],output_ids=generated,reason=aborted,replayed_deterministically_on_resume=True)
                        break
                    content=generated[:-1] if generated and generated[-1] in eos else generated
                    raw=codec.decode(generated,skip_special_tokens=False);response=codec.decode(content,skip_special_tokens=False)
                    score=grade(response,row);assert 0<=score<=100
                    record=dict(row,output_ids=generated,raw=raw,response=response,score=score,stop=stop,unused_token_ids=[i for i in generated if codec.id_to_token(i) is None],prefill_seconds=prefill,seconds=time.perf_counter()-start)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(record);completed+=1
                    meta['completed_items']=completed;save_meta(meta_path,meta);print(now(),args.stage,completed,len(rows),flush=True)
                else:meta['status']='complete'
        except BaseException as error:
            meta.update(status='error',error_type=type(error).__name__);raise
        finally:
            meta.update(ended=now(),completed_items=completed,full_mac_cohort_generated=meta['status']=='complete' and completed==len(rows),author_summary=summary(records) if meta['status']=='complete' and completed==len(rows) else None,predictions_sha256=digest(out),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            meta['events'][-1].update(end=now(),ending_items=completed,status=meta['status']);save_meta(meta_path,meta)


if __name__=='__main__':main()
