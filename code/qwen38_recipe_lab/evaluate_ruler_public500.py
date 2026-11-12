"""Bounded streaming RULER author500 evaluation, never loading the full corpus into RAM."""
import argparse,fcntl,json,platform,resource,time
from datetime import datetime,timedelta
from pathlib import Path
import torch
from tokenizers import Tokenizer
from evaluate_batched import stop_requests,system_pressure
from evaluate_suite import now,save_meta
from model import Model,Config
from prefill_inference import PrefillInference
from shared_kv_inference import SharedKVDenseInference
from ruler_tasks import digest,configurations,metrics
from ruler_evaluation import scorer,postprocessor
from ruler_public500_repair import frozen,rows,PROFILE,ROOT


def verify(record,row,codec,eos,grade):
    assert {key:record[key] for key in row}==row
    ids=record['output_ids'];assert 0<len(ids)<=row['tokens'] and not any(i in eos for i in ids[:-1])
    assert record['raw']==codec.decode(ids,skip_special_tokens=False)
    assert (record['stop']=='eos' and ids[-1] in eos) or (record['stop']=='max_new_tokens' and len(ids)==row['tokens'] and ids[-1] not in eos)
    assert record['response']==codec.decode(ids[:-1] if ids[-1] in eos else ids,skip_special_tokens=False)
    assert record['score']==grade(record['response'],row)
    assert record['unused_token_ids']==[i for i in ids if codec.id_to_token(i) is None]


def compact(record):return dict(context=record['context'],task=record['task'],kind=record['kind'],response=record['response'],outputs=record['author_row']['outputs'])


def summary(records):
    process=postprocessor();functions=metrics();scores={}
    for length in (4096,8192):
        for task,config in configurations().items():
            selected=[r for r in records if r['context']==length and r['task']==task];assert len(selected)==500
            scores[f'{length}/{task}']=functions[config['task']]['metric_fn']([process(r['response'],{}) for r in selected],[r['outputs'] for r in selected])
    return dict(task_scores_percent=scores,configuration_macro_percent={str(length):sum(v for key,v in scores.items() if key.startswith(str(length)+'/'))/13 for length in (4096,8192)})


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['sparse','dense'],required=True);p.add_argument('--out',required=True);p.add_argument('--deadline',required=True);p.add_argument('--seconds',type=int,default=1180);p.add_argument('--resume',action='store_true');a=p.parse_args();tick=time.monotonic();seconds=min(a.seconds,1180);deadline=datetime.fromisoformat(a.deadline)
    assert deadline.tzinfo is not None and 1<=a.seconds<=1200
    assert datetime.now().astimezone()<deadline-timedelta(seconds=200) and system_pressure()==1
    prepared=frozen();profile=json.loads(PROFILE.read_text());assert digest(profile['checkpoint'])==profile['checkpoint_sha256']
    sources=['evaluate_ruler_public500.py','ruler_public500.py','ruler_public500_repair.py','profiles/ruler_public500_repair.json','ruler_tasks.py','ruler_evaluation.py','shared_kv_inference.py','model.py','optim.py','polar_coefficients.json','train.py','inference.py','prefill_inference.py','reference.py','evaluate_batched.py','evaluate_suite.py','evaluate.py','eval_tasks.py','sources.lock.json','ruler_sources.lock.json','ruler_eval_sources.lock.json',str(PROFILE),'.cache/references/modeling_qwen4_exp.py','.cache/references/tokenizer.json','.cache/references/generation_config.json']
    identity=dict(suite='ruler_public500_qa_repair',checkpoint_sha256=profile['checkpoint_sha256'],prepared_meta_sha256=digest(ROOT/'meta.json'),stage=a.stage,contexts=[4096,8192],samples_per_configuration_per_length=500,target_items=13000,generation_budgets=profile['generation_budgets'],threads=1,device='cpu',greedy=True,batch_size=1,prefill_chunk=128,gdn_scan='chunk',grouped_gqa=a.stage=='sparse',dense_shared_kv=a.stage=='dense',streaming=True,sources={name:digest(name) for name in sources})
    source_lock=json.loads(Path('sources.lock.json').read_text())
    for filename in ('tokenizer.json','generation_config.json'):assert digest(Path('.cache/references')/filename)==next(item['sha256'] for item in source_lock if item['name']==filename)
    codec=Tokenizer.from_file('.cache/references/tokenizer.json');eos=set(json.loads(Path('.cache/references/generation_config.json').read_text())['eos_token_id']);grade=scorer();iterator=iter(rows(prepared,codec));out=Path(a.out);meta_path=Path(str(out)+'.meta.json')
    with Path(str(out)+'.lock').open('a') as lock,stop_requests() as requested:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);completed=0;records=[]
        if a.resume:
            meta=json.loads(meta_path.read_text());assert meta['identity']==identity and meta['status'] in ('time_budget','interrupted_signal','memory_pressure')
            with out.open() as stream:
                for line in stream:
                    record=json.loads(line);row=next(iterator);verify(record,row,codec,eos,grade);records.append(compact(record));completed+=1
            assert completed==meta['completed_items']
            if 'unscored_incomplete_row' in meta:meta.setdefault('incomplete_row_history',[]).append(meta.pop('unscored_incomplete_row'))
        else:
            assert time.monotonic()-tick<seconds,'Preflight exhausted call budget before output'
            assert not out.exists() and not meta_path.exists();out.touch(exist_ok=False);meta=dict(identity=identity,started=now(),events=[])
        assert time.monotonic()-tick<seconds,'Preflight exhausted call budget before model'
        torch.set_num_threads(1);torch.manual_seed(42);saved=torch.load(profile['checkpoint'],map_location='cpu',weights_only=True,mmap=True);assert saved['step']==saved['args']['steps']==8000
        assert saved['source_hashes']=={name:identity['sources'][name] for name in ('model.py','optim.py','polar_coefficients.json','train.py')};saved.pop('optimizers',None);model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved.pop('model'),assign=True);del saved
        runner=PrefillInference(model,'sparse',prefill_chunk=128,grouped_gqa=True) if a.stage=='sparse' else SharedKVDenseInference(model,prefill_chunk=128)
        meta.update(status='running',completed_items=completed,target_items=13000,full_paper_protocol=False,scope='Public default500 samples per13 configurations at Mac4K/8K, not original128K–1M/private Qwen harness',environment=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform()));meta['events'].append(dict(start=now(),starting_items=completed,resume=a.resume,budget_seconds=seconds));save_meta(meta_path,meta)
        def stop():
            if requested:return 'interrupted_signal'
            if system_pressure()>=2:return 'memory_pressure'
            if time.monotonic()-tick>=seconds or datetime.now().astimezone()>=deadline-timedelta(seconds=180):return 'time_budget'
        try:
            with out.open('a') as stream:
                for row in iterator:
                    reason=stop()
                    if reason:meta['status']=reason;break
                    started=time.perf_counter();cache={};last=runner(torch.tensor([row['input_ids']]),cache);prefill=time.perf_counter()-started;generated=[];ending='max_new_tokens';aborted=None
                    for step in range(row['tokens']):
                        if step%32==0 and (aborted:=stop()):break
                        assert torch.isfinite(last).all();token=int(last.argmax(-1));generated.append(token)
                        if token in eos:ending='eos';break
                        if step+1<row['tokens']:last=runner(torch.tensor([[token]]),cache)
                    if aborted:
                        meta.update(status=aborted,unscored_incomplete_row=dict(source_id=row['source_id'],output_ids=generated,reason=aborted));break
                    raw=codec.decode(generated,skip_special_tokens=False);response=codec.decode(generated[:-1] if generated[-1] in eos else generated,skip_special_tokens=False)
                    record=dict(row,output_ids=generated,raw=raw,response=response,stop=ending,score=grade(response,row),unused_token_ids=[i for i in generated if codec.id_to_token(i) is None],prefill_seconds=prefill,seconds=time.perf_counter()-started)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(compact(record));completed+=1;meta['completed_items']=completed;save_meta(meta_path,meta);print(now(),a.stage,completed,13000,flush=True)
                else:assert completed==13000;meta['status']='complete'
        except BaseException as error:meta.update(status='error',error_type=type(error).__name__);raise
        finally:
            meta.update(ended=now(),completed_items=completed,full_mac_cohort_generated=meta['status']=='complete' and completed==13000,author_summary=summary(records) if meta['status']=='complete' and completed==13000 else None,predictions_sha256=digest(out),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);meta['events'][-1].update(end=now(),ending_items=completed,status=meta['status']);save_meta(meta_path,meta)


if __name__=='__main__':main()
